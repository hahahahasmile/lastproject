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
    adjust_pct_by_price_level, make_entry_at, make_sl_tp,
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



# ---------------------------
# UI - 상단 설정
# ---------------------------
st.subheader("설정")
colA, colB, colC = st.columns(3)

with colA:
    sim_mode = st.radio("모드", ["NOW", "ROLLING", "LIVE"], index=0, horizontal=True)

sim_engine = "DTW"
w_dtw = 0.5
thr = 3.0
ratio_min = 1.5
entry_rule = "다음봉 시가"

sltp_method = "ATR"
k_sl = 1.0
k_tp = 3.0
sl_pct = -0.015
tp_pct = 0.03

fee_entry  = 0.0004
fee_exit   = 0.0005
slip_entry = 0.0003
slip_exit  = 0.0005

equity = 1000.0
risk_pct = 0.02
fast = True
max_leverage = 10.0
max_notional = 100_000.0
qty_step = 0.001
tick_size = 0.1

# ---- NOW/ROLLING: 기존 전체 UI 노출 ----
if sim_mode == "ROLLING":
    # 기존 설정 UI는 ROLLING에서만 보인다
    with colA:
        sim_engine = st.selectbox("유사도 방식", ["DTW", "Frechet", "Hybrid"], index=0)
        w_dtw = st.slider("Hybrid: DTW 가중치", 0.0, 1.0, 0.5, 0.05)
        thr = st.slider("신호 임계치(%)", 1.0, 10.0, 3.0, 0.5)
        ratio_min = st.slider("가격보정 임계배수 (현재/과거 ≥)", 1.0, 3.0, 1.5, 0.1)
        entry_rule = st.selectbox(
            "엔트리 가격 규칙",
            ["현재 로직(진입봉 종가)", "다음봉 시가", "OHLC 평균(보수적)"],
            index=1
        )
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
        max_notional = st.number_input("최대 노출 금액(USDT)", 0.0, 1_000_000.0, 100_000.0, 1000.0)
        qty_step = st.number_input("수량 스텝(계약 최소단위)", 0.0001, 1.0, 0.001, 0.0001)
        tick_size = st.number_input("호가 단위(틱 사이즈)", 0.01, 10.0, 0.1, 0.01)

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
train_end_ts = ROLL_START if (sim_mode == "ROLLING") else pred_start

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

def decide_from_future_path(hist_pct: np.ndarray, L_prefix: int, thr_pct: float):
    """프리픽스 이후의 예상 변동폭으로 LONG/SHORT/HOLD 결정"""
    if L_prefix < 1 or L_prefix >= len(hist_pct):
        return "HOLD", 0.0, 0.0
    base_now = float(hist_pct[L_prefix-1])
    future = hist_pct[L_prefix-1:] - base_now
    max_up = float(np.max(future)) if future.size else 0.0
    max_down = float(np.min(future)) if future.size else 0.0
    if max_up >= thr_pct or max_down <= -thr_pct:
        side = "LONG" if abs(max_up) >= abs(max_down) else "SHORT"
    else:
        side = "HOLD"
    return side, max_up, max_down

def show_decision_logs():
    if "decision_logs" in st.session_state and st.session_state["decision_logs"]:
        df_decisions = pd.DataFrame(st.session_state["decision_logs"])
        st.markdown("### 🧾 결정 근거 로그")
        st.dataframe(df_decisions, use_container_width=True)
        st.download_button(
            label="📥 결정 근거 로그 CSV 다운로드",
            data=df_decisions.to_csv(index=False).encode("utf-8"),
            file_name="decision_log.csv",
            mime="text/csv",
            key="dl_decision"
        )

# === NOW 리뉴얼 보조함수 ===
def _make_percent_table_from_block(df_block: pd.DataFrame) -> pd.DataFrame:
    """블록(예: 72h, 18봉) 내부의 각 4h봉을, 블록의 0~4h 시가를 앵커로 하여 퍼센트 변화율로 표시"""
    O_anchor = float(df_block['open'].iloc[0])
    out = pd.DataFrame({
        "k": np.arange(len(df_block), dtype=int),
        "r_open_%":  (df_block['open']  / O_anchor - 1.0) * 100.0,
        "r_close_%": (df_block['close'] / O_anchor - 1.0) * 100.0,
        "r_high_%":  (df_block['high']  / O_anchor - 1.0) * 100.0,
        "r_low_%":   (df_block['low']   / O_anchor - 1.0) * 100.0,
    })
    return out

def _scan_segments_28_72(df_block_pred: pd.DataFrame, thr_abs_pct: float = 2.0):
    """
    과거 '예측 블록(다음 72h)' 내부에서 진입·청산 조합 스캔.
    - 진입: 블록 시가만, 인덱스 {7,8,9,10,11,12,13,14}  (28~60h)
    - 청산: 진입 블록 종가 + 다음 3블록 종가 => 최대 4개
    반환:
      scan_df: 모든 32개 조합 (entry_k, exit_k, delta_pct)
      best:    절댓값이 가장 큰 조합 dict( entry_k, exit_k, delta_pct, side, ok )
    """
    n = len(df_block_pred)
    opens  = df_block_pred['open'].to_numpy(dtype=float)
    closes = df_block_pred['close'].to_numpy(dtype=float)
    highs  = df_block_pred['high'].to_numpy(dtype=float)
    lows   = df_block_pred['low'].to_numpy(dtype=float)

    ENTRY_KS = [7,8,9,10,11,12,13,14]  # 28~60h
    rows = []
    best = None

    for k_entry in ENTRY_KS:
        if k_entry < 0 or k_entry >= n: 
            continue
        entry_open = opens[k_entry]
        for k_exit in [k_entry, k_entry+1, k_entry+2, k_entry+3]:
            if k_exit >= n: 
                continue
            delta = (closes[k_exit] / entry_open - 1.0) * 100.0
            rows.append({"entry_k": k_entry, "exit_k": k_exit, "delta_pct": delta})
            if (best is None) or (abs(delta) > abs(best["delta_pct"])):
                best = {"entry_k": k_entry, "exit_k": k_exit, "delta_pct": float(delta)}

    scan_df = pd.DataFrame(rows).sort_values(["entry_k","exit_k"]).reset_index(drop=True)
    scan_df = scan_df.sort_values(
    "delta_pct",
    key=lambda s: s.abs(),   
    ascending=False
    ).reset_index(drop=True)
    if best is None:
        return scan_df, {"side":"HOLD","ok":False}

    side = "LONG" if best["delta_pct"] >= 0.0 else "SHORT"
    ok = (abs(best["delta_pct"]) >= thr_abs_pct)
    best.update({"side":side, "ok":bool(ok)})

    # 부가: p_entry% / p_peak%(롱) 또는 p_trough%(숏) 계산을 위해 high/low 슬라이스도 같이 계산
    O_anchor_past = float(df_block_pred['open'].iloc[0])
    k0, k1 = best["entry_k"], best["exit_k"]
    p_entry = (opens[k0] / O_anchor_past - 1.0) * 100.0
    if side == "LONG":
        p_peak = (np.max(highs[k0:k1+1]) / O_anchor_past - 1.0) * 100.0
        best.update({"p_entry_pct": float(p_entry), "p_extreme_pct": float(p_peak)})
    else:
        p_trough = (np.min(lows[k0:k1+1]) / O_anchor_past - 1.0) * 100.0
        best.update({"p_entry_pct": float(p_entry), "p_extreme_pct": float(p_trough)})

    return scan_df, best

# 공통 상태
df_log = None
topN = 5 if fast else 10
exd  = 10 if fast else 5
stepTD = pd.Timedelta(hours=step_hours)
delayTD = pd.Timedelta(hours=ENTRY_DELAY_HOURS)
st.session_state["decision_logs"] = []

# ================= NOW =================
if sim_mode == "NOW":
    st.subheader("NOW: 유사 과거 1개 선택 → 28~72h 전 범위 스캔(4시간 블록)")

    # 1) 유사 과거 후보
    cands = get_candidates(df_full, (ref_start, ref_end), ex_margin_days=exd, topN=topN, past_only=False)
    cands = [c for c in cands if c.get("sim", 0.0) >= 0.5]

    if not cands:
        st.warning("유사도 ≥ 0.75 인 과거 후보가 없습니다.")
        st.stop()

    cand = cands[0]  # 가장 유사한 1개
    st.markdown(f"- 선택된 과거 블록: **{cand['start']} ~ {cand['end']}** (sim={cand['sim']:.3f})")

    # 2) 과거 예측 블록(다음 72h)과 현재 예측 블록(금번 72h)
    past_pred_start = cand["end"]
    past_pred_end   = past_pred_start + stepTD
    df_past_pred = df_full[(df_full["timestamp"] >= past_pred_start) & (df_full["timestamp"] < past_pred_end)].reset_index(drop=True)

    cur_pred_start = pred_start
    cur_pred_end   = pred_end
    df_now_pred = df_full[(df_full["timestamp"] >= cur_pred_start) & (df_full["timestamp"] < cur_pred_end)].reset_index(drop=True)

    if len(df_past_pred) < window_size or len(df_now_pred) == 0:
        st.error("예측 블록 데이터가 부족합니다.")
        st.stop()

    # 3) 퍼센트 표(앵커 = 각 블록의 0~4h 시가)
    past_pct_tbl = _make_percent_table_from_block(df_past_pred)
    now_pct_tbl  = _make_percent_table_from_block(df_now_pred)

    # 현재 표는 '완료된 봉까지만' 채우고, 미래 봉은 공백
    # (데이터프레임 행수를 18로 맞추고, 부족분은 NaN 유지)
    if len(now_pct_tbl) < window_size:
        pad = pd.DataFrame({
            "k": np.arange(len(now_pct_tbl), window_size, dtype=int),
            "r_open_%":  np.nan, "r_close_%": np.nan, "r_high_%": np.nan, "r_low_%": np.nan
        })
        now_pct_tbl = pd.concat([now_pct_tbl, pad], ignore_index=True)

    with st.expander("📊 과거_퍼센트표 (앵커=과거 0~4h 시가)", expanded=False):
        st.dataframe(past_pct_tbl, use_container_width=True)
    with st.expander("📊 현재_퍼센트표 (앵커=현재 0~4h 시가, 미완료 봉=공백)", expanded=False):
        st.dataframe(now_pct_tbl, use_container_width=True)

    # 4) 28~72 전 범위 스캔 (진입=시가, 청산=해당+다음3 블록 종가, 총 32개)
    scan_df, best = _scan_segments_28_72(df_past_pred, thr_abs_pct=2.0)

    with st.expander("🧮 스캔요약표 (8개 진입 × 4개 청산 = 최대 32개)", expanded=True):
        st.dataframe(scan_df, use_container_width=True)

    if not best.get("ok", False):
        st.info("최대 |Δ%| < 2.0% (이상 기준 미충족) → 거래 시작 조건 불만족")
        st.stop()

    # 5) 최종 선택(방향/진입·청산/Δ%) 및 현재로 환산
    entry_k, exit_k = best["entry_k"], best["exit_k"]
    side = best["side"]
    delta_pct = best["delta_pct"]
    p_entry_pct = best["p_entry_pct"]
    p_extreme_pct = best["p_extreme_pct"]  # LONG: peak, SHORT: trough

    O_anchor_now = float(df_now_pred['open'].iloc[0])
    entry_now  = O_anchor_now * (1.0 + p_entry_pct   / 100.0)
    extreme_now= O_anchor_now * (1.0 + p_extreme_pct / 100.0)

    # ATR은 '현재 예측 블록' 내 마지막 완료 봉 기준으로 사용
    atr_now = float(df_now_pred['atr'].dropna().iloc[-1]) if 'atr' in df_now_pred.columns and df_now_pred['atr'].notna().any() else np.nan
    tp, sl = (np.nan, np.nan)
    try:
        if side == "LONG":
            tp = entry_now + k_tp * atr_now
            sl = entry_now - k_sl * atr_now
        elif side == "SHORT":
            tp = entry_now - k_tp * atr_now
            sl = entry_now + k_sl * atr_now
    except Exception:
        pass

    # 6) 출력(최종선택 요약)
    st.markdown("### ✅ 최종선택")
    st.write({
        "방향": side,
        "진입블록_k": int(entry_k),
        "청산블록_k": int(exit_k),
        "Δ% (ExitClose/EntryOpen-1)*100": float(delta_pct),
        "p_entry% (앵커대비)": float(p_entry_pct),
        "p_extreme% (앵커대비)": float(p_extreme_pct),  # LONG: peak, SHORT: trough
        "Entry_now": float(entry_now),
        ("Peak_now" if side=="LONG" else "Trough_now"): float(extreme_now),
        "ATR_now": float(atr_now) if not np.isnan(atr_now) else None,
        "TP": float(tp) if not np.isnan(tp) else None,
        "SL": float(sl) if not np.isnan(sl) else None,
    })

    # 7) 시각화(한 장) — 현재 vs 과거(앵커 기준 r_close_% 라인)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(past_pct_tbl["k"], past_pct_tbl["r_close_%"], label="과거(예측블록) r_close_%", linewidth=2)
    ax.plot(now_pct_tbl["k"],  now_pct_tbl["r_close_%"],  label="현재(예측블록) r_close_%", linewidth=2, alpha=0.8)
    ax.axvline(entry_k, color="gray", linestyle="--", linewidth=1)
    ax.axvline(exit_k,  color="gray", linestyle="--", linewidth=1)
    ax.set_title("현재 vs 가장 유사한 과거 (앵커 기준 퍼센트 라인)")
    ax.grid(True, alpha=0.3); ax.legend()
    st.pyplot(fig)

# ================= LIVE (실거래) =================
elif sim_mode == "LIVE":
    st.subheader("LIVE: 실거래 (메인넷) — 최소 셋")

    # --- 전략 프리셋(내부 고정, UI 노출 X) ---
    # 필요하면 여기 숫자만 바꿔라.
    entry_rule   = "다음봉 시가"
    sltp_method  = "ATR"
    k_sl, k_tp   = 1.0, 3.0
    sl_pct, tp_pct = -0.015, 0.03  # 함수 인자 요구로 기본값 유지
    # 유사도/임계/가격보정 등 연구용 파라미터는 LIVE에 노출/사용하지 않음

    # --- 계정/거래 설정 ---
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

    # --- 신호용 블록 계산: Entry/SL/TP 산출만 사용 (주문은 버튼으로) ---
    (ref_start, ref_end), (pred_start, pred_end) = pick_blocks(df_full['timestamp'].iloc[-1], step_hours=step_hours)
    t_entry = pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS)

    # 엔트리 기준 가격 계산 (전략 로직)
    entry_time, entry_price = make_entry_at(df_full, t_entry, rule=entry_rule)
    if entry_time is not None and entry_time < t_entry:
        seg_after = df_full[df_full["timestamp"] > t_entry]
        if not seg_after.empty:
            entry_time  = seg_after["timestamp"].iloc[0]
            entry_price = float(seg_after["open"].iloc[0])

    # SL/TP 산출 (전략 로직)
    atr_ref = float(df_full.loc[df_full["timestamp"]==entry_time, "atr"].fillna(method='ffill').iloc[0]) if entry_time is not None else None
    sl, tp = make_sl_tp(entry_price, "LONG",  # 방향 무관 공통 산출 위해 placeholder, 내부에서 값만 사용
                        method=sltp_method, atr=atr_ref,
                        sl_pct=sl_pct, tp_pct=tp_pct, k_sl=k_sl, k_tp=k_tp, tick_size=0.01)

    # --- 거래소 필터 & 레버리지/마진 설정 ---
    ensure_leverage_and_margin(tclient, symbol=trade_symbol, leverage=int(leverage), cross=use_cross)
    tick_size, qty_step = get_symbol_filters(tclient, symbol=trade_symbol)

    # --- 사이즈 % → 계약수량 변환 (가용잔고 × 레버리지 × %) / 엔트리가격 ---
    avail = bals["available_balance"]
    notional = max(0.0, avail) * float(leverage) * (float(size_pct) / 100.0)
    qty_live = (notional / float(entry_price)) if entry_price else 0.0
    # 수량/가격 거래소 규격 라운딩은 주문 함수 내부에서 한 번 더 보정됨

    # --- 가격 정보 표시 ---
    st.markdown("### 📌 주문 미리보기")
    colp1, colp2, colp3, colp4 = st.columns(4)
    colp1.metric("Entry(참조)", f"{(entry_price or 0):.2f}")
    colp2.metric("SL", f"{(sl or 0):.2f}")
    colp3.metric("TP", f"{(tp or 0):.2f}")
    colp4.metric("수량(계약)", f"{qty_live:.6f}")

    # 현재 포지션 & 청산가(거래소 리포트 값)
    with st.expander("📈 현재 포지션", expanded=True):
        try:
            infos = tclient.futures_position_information(symbol=trade_symbol)
            df_pos = pd.DataFrame(infos)
            # 표시용 최소 컬럼 추림
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

    for j, bp in enumerate(range(start_idx, len(blocks_all))):
        ref_b = blocks_all[bp-1]; pred_b = blocks_all[bp]
        cands = get_candidates(df_roll, (ref_b["start"], ref_b["end"]), ex_margin_days=exd, topN=topN, past_only=True)
        if not cands:
            pbar.progress(int(100*(j+1)/total)); continue

        cur_bp_price = float(df_roll.loc[df_roll["timestamp"] >= pred_b["start"], "close"].iloc[0])
        results=[]
        for f in cands:
            next_start=f["end"]; next_end=next_start+pd.Timedelta(hours=step_hours)
            df_next = df_roll[(df_roll["timestamp"] >= next_start) & (df_roll["timestamp"] < next_end)]
            if len(df_next) < window_size:
                continue
            closes=df_next["close"].to_numpy(); base=float(closes[0])
            pct=(closes-base)/base*100.0
            pct_adj=adjust_pct_by_price_level(cur_bp_price, base, pct, ratio_min=ratio_min)
            results.append({"sim":f["sim"], "next_start":next_start, "next_end":next_end, "pct":pct_adj})
        if not results:
            pbar.progress(int(100*(j+1)/total)); continue

        t_entry = pred_b["start"] + pd.Timedelta(hours=ENTRY_DELAY_HOURS)
        if t_entry > pred_b["end"]:
            pbar.progress(int(100*(j+1)/total)); continue

        pred_seg = df_roll[(df_roll["timestamp"] >= pred_b["start"]) & (df_roll["timestamp"] <= t_entry)]
        if len(pred_seg) == 0:
            pbar.progress(int(100*(j+1)/total)); continue
        base_cur=float(pred_seg["close"].iloc[0])
        a=((pred_seg["close"]-base_cur)/base_cur*100.0).to_numpy(dtype=float); L=len(a)

        best=None
        for r in results:
            b=np.array(r["pct"], dtype=float)[:L]
            sim_shape = 1.0 if (np.allclose(a,0) and np.allclose(b,0)) else float(cosine_similarity([a],[b])[0][0])
            if (best is None) or (sim_shape > best["sim"]):
                best={"sim":sim_shape,"flow":r}
        hist_full=np.array(best["flow"]["pct"], dtype=float)
        base_now = float(hist_full[L-1])
        fut = hist_full[L-1:] - base_now
        if L>0 and L<len(hist_full) and (np.max(fut)>=thr or np.min(fut)<=-thr):
            side = "LONG" if abs(np.max(fut)) >= abs(np.min(fut)) else "SHORT"
        else:
            side = "HOLD"
        if best["sim"] < 0.75:
            side = "HOLD"

        entry_time, entry_price = make_entry_at(df_roll, t_entry, rule=entry_rule)
        atr_ref = float(df_roll.loc[df_roll["timestamp"]==entry_time, "atr"].fillna(method='ffill').iloc[0]) if entry_time is not None else None
        sl, tp = make_sl_tp(entry_price, side, method=sltp_method,
                            atr=atr_ref, sl_pct=sl_pct, tp_pct=tp_pct,
                            k_sl=k_sl, k_tp=k_tp, tick_size=tick_size) if side!="HOLD" else (None, None)

        size = 0.0
        if side!="HOLD" and entry_price and sl:
            size = position_size(equity, risk_pct, entry_price, sl,
                                 contract_value=1.0, max_leverage=max_leverage,
                                 max_notional=max_notional, qty_step=qty_step)
        exit_time, exit_price, gross_ret, net_ret = (None, None, None, None)
        if side!="HOLD":
            exit_time, exit_price, gross_ret, net_ret = simulate_trade(
                df_roll, t_entry, pred_b["end"], side,
                entry_time, entry_price, sl, tp,
                fee_entry=fee_entry, fee_exit=fee_exit,
                slip_entry=slip_entry, slip_exit=slip_exit
            )
        trade_logs.append({
            "pred_start": pred_b["start"], "pred_end": pred_b["end"], "t_entry": t_entry,
            "side": side, "sim_prefix": best["sim"],
            "entry_time": entry_time, "entry": entry_price,
            "SL": sl, "TP": tp, "size_notional": size,
            "exit_time": exit_time, "exit": exit_price,
            "gross_ret_%": gross_ret, "net_ret_%": net_ret
        })
        pbar.progress(int(100*(j+1)/total))

    if not trade_logs:
        st.info("ROLLING 결과 없음")
        st.stop()

    df_log = pd.DataFrame(trade_logs).sort_values("pred_start").reset_index(drop=True)
    st.markdown("### 결과 테이블")
    st.dataframe(df_log)
    st.session_state["decision_logs"] += trade_logs

# -------------------------------
# 에쿼티 커브 & 카드 (NOW 모드: Sharpe/Equity 미표시)
# -------------------------------
if 'df_log' in locals() and df_log is not None and not df_log.empty:
    dates, equity_curve = build_equity_curve(df_log, equity)
    metrics = calc_metrics(df_log, equity_curve)

    if sim_mode == "NOW":
        col1, col2, col3, col5 = st.columns(4)
        col1.metric("거래수", metrics["n_trades"])
        col2.metric("Hit-rate", f"{metrics['hit_rate']:.1f}%")
        col3.metric("Avg Win/Loss", f"{metrics['avg_win']:.2f}% / {metrics['avg_loss']:.2f}%")
        col5.metric("MDD / MAR", f"{metrics['mdd']*100:.2f}% / {metrics['mar']:.2f}")
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
            st.warning(f"에쿼티 커브를 그릴 수 없습니다. lengths: dates={len(dates)}, equity={len(equity_curve)}")
else:
    st.info("아직 거래 로그가 없습니다. (조건 미충족/HOLD 등)")
