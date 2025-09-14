# ui/app.py
# pip install streamlit python-binance scikit-learn matplotlib pandas numpy python-dotenv

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from connectors import (
    connect_binance, connect_binance_trade,
    get_futures_balances, get_futures_positions,
    ensure_leverage_and_margin, get_symbol_filters,
    get_spot_balances
)
from data_fetch import fetch_futures_4h_klines, fetch_funding_rate
from features import (
    add_features, apply_static_zscore, finalize_preprocessed,
    window_is_finite, window_vector, GLOBAL_Z_COLS, FEAT_COLS,
)
from similarity import sim_tier3
from blocks import pick_blocks, enumerate_blocks
from trading_utils import (
    make_entry_at, make_sl_tp,
    position_size, simulate_trade, place_futures_market_bracket,
)
from backtest_utils import build_equity_curve, calc_metrics
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="BTC 전략 분석 (Tier3 실전 백테스트+LIVE)", page_icon="📊", layout="wide")
st.title("📈 유사 흐름 기반 BTC · NOW / ROLLING / LIVE")

# ---------------------------
# 공통 설정
# ---------------------------
step_hours = 72
window_size = 18
ENTRY_DELAY_HOURS = 28
ENTRY_RULE_FIXED = "다음봉 시가"  # 엔트리 가격 규칙 고정 (ROLLING/LIVE 내부에서만 사용)

# (ROLLING 전용) 전략 분기 임계치
LO_THR = 1.0  # 1.0% 이상일 때 거래 고려
HI_THR = 3.0   # 3.0% 이상이면 기존 전략, 그 미만(≥1.5)은 새 전략

# ---------------------------
# UI - 상단 설정
# ---------------------------
st.subheader("설정")
colA, colB, colC = st.columns(3)

with colA:
    sim_mode = st.radio("모드", ["NOW", "ROLLING", "LIVE"], index=0, horizontal=True)

# 기본 프리셋(ROLLING에서만 조절)
sim_engine = "DTW"
w_dtw = 0.5

sltp_method = "ATR"
k_sl = 1.0
k_tp = 3.0
sl_pct = -0.015
tp_pct = 0.03

fee_entry = 0.0004
fee_exit = 0.0005
slip_entry = 0.0003
slip_exit = 0.0005

equity = 1000.0
risk_pct = 0.02
fast = True
max_leverage = 10.0

# ---- ROLLING에서만 조절 가능한 옵션 ----
if sim_mode == "ROLLING":
    with colA:
        sim_engine = st.selectbox("유사도 방식", ["DTW", "Frechet", "Hybrid"], index=0)
        w_dtw = st.slider("Hybrid: DTW 가중치", 0.0, 1.0, 0.5, 0.05)
        st.caption(f"전략 기준: |변동|≥{HI_THR:.1f}% → 기존 / {LO_THR:.1f}%~<{HI_THR:.1f}% → 새 전략 / <{LO_THR:.1f}% → HOLD")
    with colB:
        sltp_method = st.radio("SL/TP 방식", ["ATR", "FIXED%"], index=0)
        if sltp_method == "ATR":
            k_sl = st.number_input("SL × ATR", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            k_tp = st.number_input("TP × ATR", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
            sl_pct = -0.015; tp_pct = 0.03
        else:
            sl_pct = st.number_input("SL % (음수)", min_value=-20.0, max_value=0.0, value=-1.5, step=0.1)/100.0
            tp_pct = st.number_input("TP %",     min_value=0.0, max_value=50.0, value=3.0, step=0.1)/100.0
            k_sl = 1.0; k_tp = 2.0
        fee_entry  = st.number_input("진입 수수료(%)", 0.0, 1.0, 0.04, 0.01) / 100.0
        fee_exit   = st.number_input("청산 수수료(%)", 0.0, 1.0, 0.05, 0.01) / 100.0
        slip_entry = st.number_input("진입 슬리피지(%)", 0.0, 0.5, 0.03, 0.01) / 100.0
        slip_exit  = st.number_input("청산 슬리피지(%)", 0.0, 0.5, 0.05, 0.01) / 100.0
    with colC:
        equity = st.number_input("가상 Equity (USDT)", min_value=10.0, value=1000.0, step=10.0)
        risk_pct = st.number_input("포지션 리스크 %", min_value=0.1, max_value=10.0, value=2.0, step=0.1)/100.0
        fast = st.checkbox("빠른 모드(TopN 줄이기, 후보 축소)", value=True)
        max_leverage = st.number_input("최대 레버리지(x)", 1.0, 50.0, 10.0, 1.0)

# ---------------------------
# 데이터 로드 & 전처리
# ---------------------------
st.caption("데이터 로드 중…")
client = connect_binance()
df_full = fetch_futures_4h_klines(client, start_time="2020-01-01")
df_funding = fetch_funding_rate(client, start_time="2020-01-01")
df_full = add_features(df_full, df_funding)  # funding만 사용

now_ts = df_full["timestamp"].iloc[-1]
(ref_start, ref_end), (pred_start, pred_end) = pick_blocks(now_ts, step_hours=step_hours)

ROLL_START = pd.Timestamp("2025-01-01 00:00:00")
train_end_ts = ROLL_START

df_full = apply_static_zscore(df_full, GLOBAL_Z_COLS, train_end_ts)
df_full = finalize_preprocessed(df_full, window_size)
if len(df_full) < window_size:
    st.error("데이터가 부족합니다.")
    st.stop()

# ---------------------------
# 공용 함수
# ---------------------------
def get_candidates(df, ref_range, ex_margin_days=5, topN=10, past_only=False):
    """참고 블록(ref_range) 기준 후보 블록 상위 topN 추출"""
    ref_seg = df[(df["timestamp"] >= ref_range[0]) & (df["timestamp"] < ref_range[1])]
    if len(ref_seg) < window_size:
        return []
    wL = ref_seg.iloc[:window_size]
    if not window_is_finite(wL):
        return []
    vec_ref = window_vector(wL, L=window_size)

    blocks = enumerate_blocks(df, step_hours=step_hours, window_size=window_size)
    ex_margin = pd.Timedelta(days=ex_margin_days)
    F = len(FEAT_COLS)
    cand = []
    for b in blocks:
        if past_only:
            if not (b["end"] <= ref_range[0] - ex_margin):
                continue
        else:
            if not ((b["end"] <= ref_range[0] - ex_margin) or (b["start"] >= ref_range[1] + ex_margin)):
                continue

        w = df[(df["timestamp"] >= b["start"]) & (df["timestamp"] < b["end"])]
        if len(w) < window_size:
            continue
        wL2 = w.iloc[:window_size]
        if not window_is_finite(wL2):
            continue
        vec_hist = window_vector(wL2, L=window_size)
        sim = sim_tier3(vec_ref, vec_hist, L=window_size, F=F, mode=sim_engine, w_dtw=w_dtw)
        cand.append({"start": b["start"], "end": b["end"], "sim": sim})
    cand.sort(key=lambda x: x["sim"], reverse=True)
    return cand[:topN]

# 새 전략: 진입가 보정 규칙(퍼센트 포인트 차감)
def _adjust_magnitude(pct_mag: float) -> float:
    """
    0.0~0.5 → 0.3p 줄임
    0.5~0.8 → 0.4p 줄임
    ≥0.8    → 0.6p 줄임
    """
    if pct_mag < 0.5:
        return max(0.0, pct_mag - 0.1)
    elif pct_mag < 0.8:
        return max(0.0, pct_mag - 0.2)
    else:
        return max(0.0, pct_mag - 0.3)

def _get_close_at_or_next(df: pd.DataFrame, ts: pd.Timestamp):
    """정확히 ts에 close가 없으면 다음 봉 close 사용"""
    row = df[df["timestamp"] == ts]
    if not row.empty:
        return float(row["close"].iloc[0])
    seg_after = df[df["timestamp"] > ts]
    if not seg_after.empty:
        return float(seg_after["close"].iloc[0])
    return None

def _touch_entry(df: pd.DataFrame, start_ts, end_ts, side: str, target_price: float):
    """start_ts~end_ts 구간에서 리밋가 터치 여부 확인"""
    seg = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    if seg.empty:
        return None, None
    if side == "LONG":
        hit = seg[seg["low"] <= target_price]
        if hit.empty: return None, None
        return hit["timestamp"].iloc[0], float(target_price)
    else:
        hit = seg[seg["high"] >= target_price]
        if hit.empty: return None, None
        return hit["timestamp"].iloc[0], float(target_price)

# 명시적 리스크 사이징(레버리지 캡 적용)
def _risk_size_with_leverage(entry_price, sl, eq_run, risk_pct, max_leverage):
    # ★ cap/리스크예산 전부 무시하고, "입력한 레버리지 그대로" 강제
    if eq_run is None or eq_run <= 0:
        return 0.0, 0.0, False
    size_notional = float(eq_run) * float(max_leverage)  # = equity × 입력 레버리지
    used_lev = float(max_leverage)                       # 표기/에쿼티 갱신용 배율도 동일
    cap_hit = False                                      # cap 개념 제거
    return size_notional, used_lev, cap_hit

# 공통 상태
df_log = None
topN = 5 if fast else 10
exd  = 10 if fast else 5
stepTD = pd.Timedelta(hours=step_hours)
delayTD = pd.Timedelta(hours=ENTRY_DELAY_HOURS)
st.session_state["decision_logs"] = []

# ================= NOW =================
if sim_mode == "NOW":
    st.subheader("NOW: 28h 지연 엔트리 · 1회 거래")

    # 1) 과거 후보 탐색 (그래프 비교용)
    cands = get_candidates(df_full, (ref_start, ref_end), ex_margin_days=exd, topN=topN, past_only=False)
    results = []
    for f in cands:
        next_start = f["end"]; next_end = next_start + stepTD
        df_next = df_full[(df_full["timestamp"] >= next_start) & (df_full["timestamp"] < next_end)]
        if len(df_next) < window_size:
            continue
        closes = df_next["close"].to_numpy()
        base = float(df_next["open"].iloc[0])
        pct_raw = (closes - base) / base * 100.0
        results.append({"sim": f["sim"], "next_start": next_start, "next_end": next_end, "pct": pct_raw})

    # 2) 현재 진행 구간 + 프리픽스
    t_entry = pred_start + delayTD
    if now_ts < t_entry:
        st.info(f"데이터 부족: 엔트리 고려 시점({t_entry})까지 28h가 지나지 않음.")
        st.stop()

    cur_pred_seg = df_full[
        (df_full["timestamp"] >= pred_start) &
        (df_full["timestamp"] <= min(now_ts, pred_end))
    ]
    if len(cur_pred_seg) == 0 or len(results) == 0:
        st.info("데이터 부족"); st.stop()

    base_cur = float(cur_pred_seg["open"].iloc[0])   # 현재 블록 첫 시가(=앵커)
    a_plot = ((cur_pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)

    prefix_end = min(t_entry, pred_end)
    cur_prefix = cur_pred_seg[cur_pred_seg["timestamp"] <= prefix_end]
    a = ((cur_prefix["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)
    L = len(a)

    # 3) 프리픽스 유사도 기준 베스트(그래프용)
    best = None
    for r in results:
        b = np.array(r["pct"], dtype=float)[:L]
        sim_shape = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) else float(cosine_similarity([a], [b])[0][0])
        if (best is None) or (sim_shape > best["sim"]):
            best = {"sim": sim_shape, "flow": r}
    hist_full = np.array(best["flow"]["pct"], dtype=float)

    best_start = best["flow"]["next_start"]
    best_end   = best_start + stepTD
    df_best_next = df_full[(df_full["timestamp"] >= best_start) & (df_full["timestamp"] < best_end)]
    
    past_pct_tbl = pd.DataFrame({
        "k": np.arange(len(df_best_next), dtype=int),
        "r_open_%":  (df_best_next['open']  / df_best_next['open'].iloc[0] - 1.0) * 100.0,
        "r_close_%": (df_best_next['close'] / df_best_next['open'].iloc[0] - 1.0) * 100.0,
        "r_high_%":  (df_best_next['high']  / df_best_next['open'].iloc[0] - 1.0) * 100.0,
        "r_low_%":   (df_best_next['low']   / df_best_next['open'].iloc[0] - 1.0) * 100.0,
    }).reset_index(drop=True)
    
    now_pct_tbl = pd.DataFrame({
        "k": np.arange(len(cur_pred_seg), dtype=int),
        "r_open_%":  (cur_pred_seg['open']  / cur_pred_seg['open'].iloc[0] - 1.0) * 100.0,
        "r_close_%": (cur_pred_seg['close'] / cur_pred_seg['open'].iloc[0] - 1.0) * 100.0,
        "r_high_%":  (cur_pred_seg['high']  / cur_pred_seg['open'].iloc[0] - 1.0) * 100.0,
        "r_low_%":   (cur_pred_seg['low']   / cur_pred_seg['open'].iloc[0] - 1.0) * 100.0,
    }).reset_index(drop=True)
    
    with st.expander("📊 과거_퍼센트표 (앵커=과거 0~4h 시가, 원시%)", expanded=False):
        st.dataframe(past_pct_tbl, use_container_width=True)
    with st.expander("📊 현재_퍼센트표 (앵커=현재 0~4h 시가, 원시%)", expanded=False):
        st.dataframe(now_pct_tbl, use_container_width=True)
        
    st.markdown("### ⏱️ 시간 정보")
    st.write({
        "현재 블록 구간": f"{pred_start} ~ {pred_end}",
        "28h 지연 엔트리 시점": str(t_entry)
    })
    # 4) 그래프 — 진행 vs 매칭(원시%)
    fig, ax = plt.subplots(figsize=(9,3))
    ax.plot(np.arange(len(hist_full)), hist_full, label="매칭 72h(원시%)")
    ax.plot(np.arange(len(a_plot)), a_plot, label=f"현재 진행 (L={len(a_plot)})")
    ax.axvline(L-1, ls="--", label="엔트리 기준(28h)")
    # LO/HI 기준선 표시
    ax.axhline(HI_THR, ls="--"); ax.axhline(-HI_THR, ls="--")
    ax.axhline(LO_THR, ls=":");  ax.axhline(-LO_THR, ls=":")
    ax.axhline(0, ls=":")
    ax.set_title("NOW: 28h 기준 · 진행 vs 매칭 (원시%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("### 🎯 앵커 퍼센트 → 가격 & SL/TP 계산")
    colu1, colu2 = st.columns([1.4, 1])
    with colu1:
        user_pct_str = st.text_input(
            "앵커 대비 퍼센트(%)",
            value="0",
            help="0~4h 앵커(현재 블록 첫 시가) 대비 퍼센트, 예: 2.5 또는 -1.2"
        )
    with colu2:
        side_manual = st.radio("방향", ["LONG", "SHORT"], index=0, horizontal=True)
        
    if st.button("계산하기"):
        try:
            user_pct = float(user_pct_str)
        except ValueError:
            st.error("퍼센트를 숫자로 입력하세요. 예: 2.5 또는 -1.2")
        else:
            # 앵커(현재 블록 첫 시가) 기준 가격 매핑
            O_anchor_now = float(cur_pred_seg['open'].iloc[0])
            mapped_price = O_anchor_now * (1.0 + user_pct / 100.0)
            
            atr_ref = None
            if 'atr' in cur_pred_seg.columns and cur_pred_seg['atr'].notna().any():
                atr_ref = float(cur_pred_seg['atr'].dropna().iloc[-1])
                
            sl_val, tp_val = make_sl_tp(
                mapped_price,
                side_manual,
                method=sltp_method,
                atr=atr_ref,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                k_sl=k_sl,
                k_tp=k_tp,
                tick_size=0.0
            )

            st.success("계산 완료")
            st.write({
                "anchor(현재 0~4h 시가)": float(O_anchor_now),
                "입력 퍼센트(%)": float(user_pct),
                "계산된 가격": float(mapped_price),
                "방향": side_manual,
                "SL": float(sl_val) if sl_val else None,
                "TP": float(tp_val) if tp_val else None,
            })
    # (참고) 상위 후보 미니 비교
    sims = []
    for r in results:
        b = np.array(r["pct"], dtype=float)[:L]
        sim_prefix = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) else float(cosine_similarity([a],[b])[0][0])
        sims.append({
            "sim_prefix": sim_prefix,
            "next_start": r["next_start"],
            "next_end":   r["next_end"],
            "pct":        r["pct"],
        })
    topK = 5
    top_cands = sorted(sims, key=lambda x: x["sim_prefix"], reverse=True)[:min(topK, len(sims))]
    if top_cands:
        ncols = len(top_cands)
        fig_row, axes = plt.subplots(1, ncols, figsize=(3.2*ncols, 2.6), sharey=True)
        if ncols == 1: axes = [axes]
        for ax2, c in zip(axes, top_cands):
            hist_i = np.array(c["pct"], dtype=float)
            ax2.plot(np.arange(len(hist_i)), hist_i, label="과거 72h(원시%)")
            ax2.plot(np.arange(len(a_plot)), a_plot, label=f"현재 진행 (L={len(a_plot)})", linewidth=2)
            ax2.axvline(L-1, ls="--"); ax2.axhline(0, ls=":", alpha=0.7)
            ax2.axhline(HI_THR, ls="--", alpha=0.3); ax2.axhline(-HI_THR, ls="--", alpha=0.3)
            ax2.axhline(LO_THR, ls=":", alpha=0.3);  ax2.axhline(-LO_THR, ls=":", alpha=0.3)
            try:
                ttl = f"{pd.to_datetime(c['next_start']).strftime('%Y-%m-%d %H:%M')}\nsim={c['sim_prefix']:.2f}"
            except Exception:
                ttl = f"sim={c['sim_prefix']:.2f}"
            ax2.set_title(ttl, fontsize=10); ax2.grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        fig_row.suptitle("상위 후보 비교 (프리픽스 기준, 원시%)", fontsize=12, y=1.02)
        plt.tight_layout(); st.pyplot(fig_row)
    else:
        st.caption("상위 후보가 없습니다.")

# ================= LIVE (실거래) =================
elif sim_mode == "LIVE":
    st.subheader("LIVE: 실거래 (메인넷)")
    entry_rule   = ENTRY_RULE_FIXED
    sltp_method  = "ATR"
    k_sl, k_tp   = 1.0, 3.0
    sl_pct, tp_pct = -0.015, 0.03

    with st.expander("💳 계정 · 선물 지갑 (메인넷)", expanded=True):
        tclient = connect_binance_trade()
        trade_symbol  = st.text_input("거래 심볼", value="BTCUSDT")
        leverage      = st.number_input("레버리지(x)", min_value=1, max_value=100, value=10, step=1)
        margin_mode   = st.radio("마진 모드", ["교차(Cross)", "격리(Isolated)"], index=0, horizontal=True)
        use_cross     = (margin_mode == "교차(Cross)")
        size_pct = st.slider("사이즈 % (가용잔고 기준)", 0.1, 100.0, 2.0, 0.1)

        bals = get_futures_balances(tclient)
        colb1, colb2 = st.columns(2)
        colb1.metric("USDT Wallet", f"{bals['wallet_balance']:.2f}")
        colb2.metric("USDT Available", f"{bals['available_balance']:.2f}")
        
    with st.expander("💰 Spot 지갑 (현물)", expanded=False):
        try:
            spot_bals = get_spot_balances(tclient)
            if spot_bals:
                df_spot = pd.DataFrame(spot_bals)
                st.dataframe(df_spot, use_container_width=True)
            else:
                st.caption("잔고 없음")
        except Exception as e:
            st.error(f"Spot 잔고 조회 실패: {e}")

    (ref_start, ref_end), (pred_start, pred_end) = pick_blocks(df_full['timestamp'].iloc[-1], step_hours=step_hours)
    t_entry = pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS)

    entry_time, entry_price = make_entry_at(df_full, t_entry, rule=entry_rule)
    if entry_time is not None and entry_time < t_entry:
        seg_after = df_full[df_full["timestamp"] > t_entry]
        if not seg_after.empty:
            entry_time  = seg_after["timestamp"].iloc[0]
            entry_price = float(seg_after["open"].iloc[0])

    atr_ref = float(df_full.loc[df_full["timestamp"]==entry_time, "atr"].fillna(method='ffill').iloc[0]) if entry_time is not None else None
    sl, tp = make_sl_tp(entry_price, "LONG",
                        method=sltp_method, atr=atr_ref,
                        sl_pct=sl_pct, tp_pct=tp_pct, k_sl=k_sl, k_tp=k_tp, tick_size=0.0)

    ensure_leverage_and_margin(tclient, symbol=trade_symbol, leverage=int(leverage), cross=use_cross)
    tick_size, qty_step = get_symbol_filters(tclient, symbol=trade_symbol)

    avail = bals["available_balance"]
    notional = max(0.0, avail) * float(leverage) * (float(size_pct) / 100.0)
    qty_live = (notional / float(entry_price)) if entry_price else 0.0

    st.markdown("### 📌 주문 미리보기")
    colp1, colp2, colp3, colp4 = st.columns(4)
    colp1.metric("Entry(참조)", f"{(entry_price or 0):.2f}")
    colp2.metric("SL", f"{(sl or 0):.2f}")
    colp3.metric("TP", f"{(tp or 0):.2f}")
    colp4.metric("수량(계약)", f"{qty_live:.6f}")

    with st.expander("📈 현재 포지션", expanded=True):
        try:
            infos = tclient.futures_position_information(symbol=trade_symbol)
            df_pos = pd.DataFrame(infos)
            keep = ["symbol","positionAmt","entryPrice","unRealizedProfit","leverage","marginType","liquidationPrice"]
            show_cols = [c for c in keep if c in df_pos.columns]
            if not df_pos.empty:
                st.dataframe(df_pos[show_cols], use_container_width=True)
            else:
                st.caption("보유 포지션 없음")
        except Exception as e:
            st.error(f"포지션 조회 실패: {e}")

    st.warning("⚠️ 실제 주문이 전송됩니다. 소액으로 테스트하세요.")
    colbtn1, colbtn2 = st.columns(2)

    def _place(side_label: str):
        side = "LONG" if side_label == "LONG" else "SHORT"
        try:
            od = place_futures_market_bracket(
                client=tclient, symbol=trade_symbol, side=side,
                qty=float(qty_live), entry_price_ref=float(entry_price),
                sl_price=float(sl), tp_price=float(tp),
                qty_step=qty_step or 0.0, tick_size=tick_size or 0.0
            )
            st.success(f"{side_label} 주문 접수: orderId={od.get('orderId')}")
        except Exception as e:
            st.error(f"{side_label} 주문 실패: {e}")

    if colbtn1.button("🚀 Buy / Long"):
        _place("LONG")
    if colbtn2.button("🚀 Sell / Short"):
        _place("SHORT")

# ================ ROLLING =================
else:
    st.subheader("ROLLING: 28h 지연 엔트리 · 블록당 1회 거래 백테스트")
    ROLL_START = pd.Timestamp("2025-01-01 00:00:00")
    df_roll = df_full[df_full["timestamp"] >= (ROLL_START - pd.Timedelta(hours=72))].reset_index(drop=True)
    if len(df_roll) < window_size:
        st.warning("ROLLING: 데이터 부족")
        st.stop()

    blocks_all = enumerate_blocks(df_roll, step_hours=step_hours, window_size=window_size)
    start_idx=None
    for i in range(1, len(blocks_all)):
        if blocks_all[i]["start"] >= ROLL_START:
            start_idx=i; break
    if start_idx is None:
        st.warning("ROLLING: 2025년 이후 pred 블록 없음")
        st.stop()

    trade_logs=[]
    pbar = st.progress(0); total = (len(blocks_all)-start_idx)
    eq_run = float(equity)  # ▶ 트레이드별 자본 추적

    for j, bp in enumerate(range(start_idx, len(blocks_all))):
        ref_b = blocks_all[bp-1]; pred_b = blocks_all[bp]

        # 후보 추출(과거)
        cands = get_candidates(df_roll, (ref_b["start"], ref_b["end"]), ex_margin_days=exd, topN=topN, past_only=True)
        if not cands:
            pbar.progress(int(100*(j+1)/total)); continue

        # 각 후보의 다음 72h (종가 기준 퍼센트 시계열)
        results=[]
        for f in cands:
            next_start=f["end"]; next_end=next_start+pd.Timedelta(hours=step_hours)
            df_next = df_roll[(df_roll["timestamp"] >= next_start) & (df_roll["timestamp"] < next_end)]
            if len(df_next) < window_size:
                continue
            closes = df_next["close"].to_numpy()
            baseC   = float(closes[0])                  # 기준: 후보 28h 종가(=다음 구간 첫 close)
            pct_c   = (closes - baseC)/baseC*100.0      # 종가 기반 변화율
            results.append({
                "sim":f["sim"], "next_start":next_start, "next_end":next_end,
                "pct":pct_c, "df_next": df_next.reset_index(drop=True), "base_close": baseC
            })
        if not results:
            pbar.progress(int(100*(j+1)/total)); continue

        # 현재 pred 프리픽스(종가 기준)
        t_entry = pred_b["start"] + pd.Timedelta(hours=ENTRY_DELAY_HOURS)
        if t_entry > pred_b["end"]:
            pbar.progress(int(100)*(j+1)/total); continue

        pred_seg = df_roll[(df_roll["timestamp"] >= pred_b["start"]) & (df_roll["timestamp"] <= t_entry)]
        if len(pred_seg) == 0:
            pbar.progress(int(100*(j+1)/total)); continue
        base_cur=float(pred_seg["close"].iloc[0])
        a=((pred_seg["close"]-base_cur)/base_cur*100.0).to_numpy(dtype=float); L=len(a)

        # 프리픽스 유사도 최고 후보
        best=None
        for r in results:
            b=np.array(r["pct"], dtype=float)[:L]
            sim_shape = 1.0 if (np.allclose(a,0) and np.allclose(b,0)) else float(cosine_similarity([a],[b])[0][0])
            if (best is None) or (sim_shape > best["sim"]):
                best={"sim":sim_shape,"flow":r}

        hist_full=np.array(best["flow"]["pct"], dtype=float)
        base_now = float(hist_full[L-1])          # 후보 28h 시점 종가 변화율
        fut = hist_full[L-1:] - base_now          # 이후 구간 변화율(후행)

        # 방향/크기
        idx_max = int(np.argmax(fut))
        idx_min = int(np.argmin(fut))
        max_up = float(np.max(fut))               # 양수
        min_dn = float(np.min(fut))               # 음수

        side = "HOLD"
        if best["sim"] >= 0.75:
            mag_up = abs(max_up)
            mag_dn = abs(min_dn)
            mag = max(mag_up, mag_dn)
            if mag >= LO_THR:                     # 1.5% 이상일 때만 거래 고려
                side = "LONG" if mag_up >= mag_dn else "SHORT"

        # 전략 분기
        entry_time, entry_price, entry_target = (None, None, None)

        if side in ("LONG","SHORT"):
            if max(abs(max_up), abs(min_dn)) >= HI_THR:
                # ---- 기존 전략 (|변동| ≥ 3%): t_entry에서 즉시 진입 ----
                etime, eprice = make_entry_at(df_roll, t_entry, rule=ENTRY_RULE_FIXED)
                if etime is not None and etime < t_entry:
                    seg_after = df_roll[df_roll["timestamp"] > t_entry]
                    if not seg_after.empty:
                        etime  = seg_after["timestamp"].iloc[0]
                        eprice = float(seg_after["open"].iloc[0])
                entry_time, entry_price = etime, eprice
            else:
                # ---- 새 전략 (1.5% ≤ |변동| < 3%): 종가 기준 되돌림-매핑 리밋가 + 터치체결 ----
                df_next_best = best["flow"]["df_next"]
                base_hist_close = float(best["flow"]["base_close"])  # 후보 28h 종가
                cur_28h_close = _get_close_at_or_next(df_roll, t_entry)  # 현재 28h 종가(없으면 다음 봉 종가)

                if cur_28h_close is not None:
                    if side == "LONG":
                        end_k = min((L-1) + idx_max, len(df_next_best)-1)
                        lows_slice = df_next_best["low"].iloc[:end_k+1].to_numpy(dtype=float)
                        if lows_slice.size > 0:
                            low_min = float(np.min(lows_slice))
                            drop_pct = (low_min / base_hist_close - 1.0) * 100.0  # 음수
                            mag_adj = _adjust_magnitude(abs(drop_pct))
                            drop_adj = -mag_adj
                            entry_target = cur_28h_close * (1.0 + drop_adj/100.0)
                            entry_time, entry_price = _touch_entry(df_roll, t_entry, pred_b["end"], "LONG", entry_target)
                    else:
                        end_k = min((L-1) + idx_min, len(df_next_best)-1)
                        highs_slice = df_next_best["high"].iloc[:end_k+1].to_numpy(dtype=float)
                        if highs_slice.size > 0:
                            high_max = float(np.max(highs_slice))
                            up_pct = (high_max / base_hist_close - 1.0) * 100.0  # 양수
                            mag_adj = _adjust_magnitude(abs(up_pct))
                            up_adj = mag_adj
                            entry_target = cur_28h_close * (1.0 + up_adj/100.0)
                            entry_time, entry_price = _touch_entry(df_roll, t_entry, pred_b["end"], "SHORT", entry_target)

        # SL/TP & 사이징 & 시뮬
        atr_ref = None
        if entry_time is not None:
            row_at = df_roll[df_roll["timestamp"]==entry_time]
            if not row_at.empty and row_at["atr"].notna().any():
                atr_ref = float(row_at["atr"].fillna(method='ffill').iloc[0])

        sl, tp = (None, None)
        if side in ("LONG","SHORT") and entry_time is not None and entry_price is not None:
            sl, tp = make_sl_tp(entry_price, side, method=sltp_method,
                                atr=atr_ref, sl_pct=sl_pct, tp_pct=tp_pct,
                                k_sl=k_sl, k_tp=k_tp, tick_size=0.0)

        size = 0.0
        used_lev = 0.0
        cap_hit = False
        if side in ("LONG","SHORT") and entry_time is not None and entry_price is not None and sl:
            size, used_lev, cap_hit = _risk_size_with_leverage(entry_price, sl, eq_run, risk_pct, max_leverage)

        exit_time, exit_price, gross_ret, net_ret = (None, None, None, None)
        if side in ("LONG","SHORT") and entry_time is not None and entry_price is not None:
            exit_time, exit_price, gross_ret, net_ret = simulate_trade(
                df_roll, t_entry, pred_b["end"], side,  # 시뮬 윈도 시작은 t_entry
                entry_time, entry_price, sl, tp,
                fee_entry=fee_entry, fee_exit=fee_exit,
                slip_entry=slip_entry, slip_exit=slip_exit
            )
        else:
            if side in ("LONG","SHORT"):
                side = "HOLD"  # 미터치·미진입

        # ▶ 자본 기반 수치 업데이트 (레버리지 변경 효과가 테이블에 바로 반영됨)
        ret_pct = (net_ret or 0.0) / 100.0
        eq_before = eq_run
        pnl_usd = (size or 0.0) * ret_pct
        eq_run = eq_run + pnl_usd
        ret_equity_pct = (pnl_usd / (eq_before if eq_before > 0 else 1.0)) * 100.0

        trade_logs.append({
            "pred_start": pred_b["start"], "pred_end": pred_b["end"], "t_entry": t_entry,
            "side": side, "sim_prefix": best["sim"],
            "entry_time": entry_time, "entry": entry_price, "entry_target": entry_target,
            "SL": sl, "TP": tp, "size_notional": size, "used_lev": used_lev, "cap_hit": cap_hit,
            "exit_time": exit_time, "exit": exit_price,
            "gross_ret_%": gross_ret, "net_ret_%": net_ret,  # 계산엔 유지하되 표시에서 숨김
            "eq_before": eq_before, "eq_after": eq_run,
            "pnl_usd": pnl_usd, "ret_equity_%": ret_equity_pct,
        })
        pbar.progress(int(100*(j+1)/total))

    if not trade_logs:
        st.info("ROLLING 결과 없음")
        st.stop()

    df_log = pd.DataFrame(trade_logs).sort_values("pred_start").reset_index(drop=True)

    # ===== 표시용 테이블: 가격기준 % 숨기고 레버리지 반영 %를 메인으로 =====
    df_show = df_log.copy()
    df_show = df_show.drop(columns=["gross_ret_%", "net_ret_%"], errors="ignore")
    df_show = df_show.rename(columns={"ret_equity_%": "ret_%(levered)"})
    cols = ["pred_start","pred_end","t_entry","side","sim_prefix",
            "entry_time","entry","entry_target","SL","TP",
            "size_notional","used_lev","cap_hit","pnl_usd","ret_%(levered)",
            "eq_before","eq_after","exit_time","exit"]
    df_show = df_show[[c for c in cols if c in df_show.columns]]

    st.markdown("### 결과 테이블 (레버리지 반영 수익률)")
    st.caption("ret_%(levered) = net_ret_% × (size_notional / eq_before)")
    st.dataframe(df_show, use_container_width=True)

# -------------------------------
# 에쿼티 커브 & 카드 (NOW 모드: Sharpe/Equity 미표시)
# -------------------------------
if 'df_log' in locals() and df_log is not None and not df_log.empty:
    dates, equity_curve = build_equity_curve(df_log, float(equity))
    metrics = calc_metrics(df_log, equity_curve)

    if sim_mode == "NOW":
        pass
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("거래수", metrics["n_trades"])
        col2.metric("Hit-rate", f"{metrics['hit_rate']:.1f}%")
        col3.metric("Avg Win/Loss", f"{metrics['avg_win']:.2f}% / {metrics['avg_loss']:.2f}%")
        col4.metric("Sharpe(연율화)", f"{metrics['sharpe']:.2f}")
        col5.metric("MDD / MAR", f"{metrics['mdd']*100:.2f}% / {metrics['mar']:.2f}")

        if dates and equity_curve and (len(dates) == len(equity_curve)):
            fig, ax = plt.subplots(figsize=(10, 3.2))
            ax.plot(dates, equity_curve, linewidth=2)
            ax.set_title("Equity Curve (net)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("에쿼티 커브를 그릴 수 없습니다.")
else:
    st.info("아직 거래 로그가 없습니다. (조건 미충족/HOLD 등)")
