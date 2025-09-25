# ui/app.py (aligned A-variants)
# pip install streamlit python-binance scikit-learn matplotlib pandas numpy python-dotenv

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata

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
# Static-only
from similarity import sim_tier3
from blocks import pick_blocks, enumerate_blocks
from trading_utils import (
    make_entry_at, make_sl_tp,
    position_size, simulate_trade, place_futures_market_bracket,
)
from backtest_utils import build_equity_curve, calc_metrics
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 기본 UI 설정
# ---------------------------
st.set_page_config(page_title="BTC 전략 분석 (Tier3 실전 백테스트+LIVE)", page_icon="📊", layout="wide")
st.title("📈 유사 흐름 기반 BTC · NOW / ROLLING / LIVE / (A)")

# ---------------------------
# 공통 하이퍼파라미터
# ---------------------------
step_hours = 72
window_size = 18
ENTRY_DELAY_HOURS = 28
ENTRY_RULE_FIXED = "다음봉 시가"

LO_THR = 1.0
HI_THR = 3.0

# ---------------------------
# 상단 UI
# ---------------------------
st.subheader("설정")
colA, colB, colC = st.columns(3)

with colA:
    sim_mode = st.radio("모드", ["NOW", "ROLLING", "LIVE", "NOW (A)", "ROLLING (A)"], index=0, horizontal=True)

    # ROLLING 기본 (NOW/LIVE도 공유)
    sim_engine = "DTW"
    w_dtw = 0.5

    sltp_method = "ATR"
    k_sl = 1.0
    k_tp = 3.0
    sl_pct = -0.015
    tp_pct = 0.03

    fee_entry = 0.0004
    fee_exit  = 0.0005
    slip_entry = 0.0003
    slip_exit  = 0.0005

    equity = 1000.0
    risk_pct = 0.02
    fast = True
    max_leverage = 10.0

if sim_mode == "ROLLING":
    with colA:
        sim_engine = st.selectbox("유사도 방식", ["DTW", "Frechet", "Hybrid"], index=0)
        w_dtw = st.slider("Hybrid: DTW 가중치", 0.0, 1.0, 0.5, 0.05)
        st.caption(f"|Δ|≥{HI_THR:.1f}% → 즉시 / {LO_THR:.1f}%~<{HI_THR:.1f}% → 되돌림 / <{LO_THR:.1f}% → HOLD")
    with colB:
        sltp_method = st.radio("SL/TP 방식", ["ATR", "FIXED%"], index=0)
        if sltp_method == "ATR":
            k_sl = st.number_input("SL × ATR", 0.1, 5.0, 1.0, 0.1)
            k_tp = st.number_input("TP × ATR", 0.1, 10.0, 3.0, 0.1)
            sl_pct = -0.015; tp_pct = 0.03
        else:
            sl_pct = st.number_input("SL % (음수)", -20.0, 0.0, -1.5, 0.1) / 100.0
            tp_pct = st.number_input("TP %", 0.0, 50.0, 3.0, 0.1) / 100.0
            k_sl, k_tp = 1.0, 2.0
        fee_entry = st.number_input("진입 수수료(%)", 0.0, 1.0, 0.04, 0.01) / 100.0
        fee_exit  = st.number_input("청산 수수료(%)", 0.0, 1.0, 0.05, 0.01) / 100.0
        slip_entry = st.number_input("진입 슬리피지(%)", 0.0, 0.5, 0.03, 0.01) / 100.0
        slip_exit  = st.number_input("청산 슬리피지(%)", 0.0, 0.5, 0.05, 0.01) / 100.0
    with colC:
        equity = st.number_input("가상 Equity (USDT)", 10.0, value=1000.0, step=10.0)
        risk_pct = st.number_input("포지션 리스크 %", 0.1, 10.0, 2.0, 0.1) / 100.0
        fast = st.checkbox("빠른 모드", value=True)
        max_leverage = st.number_input("최대 레버리지(x)", 1.0, 50.0, 10.0, 1.0)

# ---------------------------
# 데이터 로드 & 전처리
# ---------------------------
st.caption("데이터 로드 중…")
client = connect_binance()
df_raw = fetch_futures_4h_klines(client, start_time="2020-01-01")
df_funding = fetch_funding_rate(client, start_time="2020-01-01")
df_feat = add_features(df_raw, df_funding)

train_end_ts_static = pd.Timestamp("2022-07-01 00:00:00")
df_full_static = apply_static_zscore(df_feat.copy(), GLOBAL_Z_COLS, train_end_ts_static)
df_full_static = finalize_preprocessed(df_full_static, window_size)
now_ts = df_full_static["timestamp"].iloc[-1]
(ref_start, ref_end), (pred_start, pred_end) = pick_blocks(now_ts, step_hours=step_hours)

if len(df_full_static) < window_size:
    st.error("데이터가 부족합니다."); st.stop()

# ---------------------------
# A-logic helpers
# ---------------------------
def _window_is_finite_a(df_window, cols):
    arr = df_window[cols].to_numpy()
    return np.isfinite(arr).all()

def _window_vector_a(df_window, feat_cols, L=18):
    X = df_window[feat_cols].to_numpy(dtype=float)
    MINMAX_COLS = ['log_ret','atr_z','vol_pct_z']
    for c in MINMAX_COLS:
        if c in feat_cols:
            j = feat_cols.index(c); v = X[:, j]
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            X[:, j] = 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.reshape(-1)

def get_candidates_a(df_pool, ref_range, df_ref, feat_cols, step_hours=72, window_size=18,
                     sim_mode='DTW', w_dtw=0.5, topN=10, ex_margin_days=5):
    ref_seg = df_ref[(df_ref["timestamp"] >= ref_range[0]) & (df_ref["timestamp"] < ref_range[1])]
    if len(ref_seg) < window_size: return []
    wL = ref_seg.iloc[:window_size]
    if not _window_is_finite_a(wL, feat_cols): return []
    vec_ref = _window_vector_a(wL, feat_cols, L=window_size)

    blocks = enumerate_blocks(df_pool, step_hours=step_hours, window_size=window_size)
    ex_margin = pd.Timedelta(days=ex_margin_days)
    F = len(feat_cols); cand = []
    for b in blocks:
        if not (b["end"] <= ref_range[0] - ex_margin or b["start"] >= ref_range[1] + ex_margin): continue
        w = df_pool[(df_pool["timestamp"] >= b["start"]) & (df_pool["timestamp"] < b["end"])]
        if len(w) < window_size: continue
        wL2 = w.iloc[:window_size]
        if not _window_is_finite_a(wL2, feat_cols): continue
        vec_hist = _window_vector_a(wL2, feat_cols, L=window_size)
        sim = sim_tier3(vec_ref, vec_hist, L=window_size, F=F, mode=sim_mode, w_dtw=w_dtw)
        cand.append({"start": b["start"], "end": b["end"], "sim": sim})
    cand.sort(key=lambda x: x["sim"], reverse=True)
    return cand[:topN]

def _adjust_magnitude_a(pct_mag: float) -> float:
    return max(0.0, pct_mag)

def _get_close_at_or_next_a(df: pd.DataFrame, ts: pd.Timestamp):
    row = df[df["timestamp"] == ts]
    if not row.empty: return float(row["close"].iloc[0])
    seg_after = df[df["timestamp"] > ts]
    if not seg_after.empty: return float(seg_after["close"].iloc[0])
    return None

def _touch_entry_a(df: pd.DataFrame, start_ts, end_ts, side: str, target_price: float):
    seg = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    if seg.empty: return None, None
    if side == "LONG":
        hit = seg[seg["low"] <= target_price]
        return (hit["timestamp"].iloc[0], float(target_price)) if not hit.empty else (None, None)
    else:
        hit = seg[seg["high"] >= target_price]
        return (hit["timestamp"].iloc[0], float(target_price)) if not hit.empty else (None, None)

# 기존 공용
def get_candidates(df, ref_range, ex_margin_days=5, topN=10, past_only=False):
    ref_seg = df[(df["timestamp"] >= ref_range[0]) & (df["timestamp"] < ref_range[1])]
    if len(ref_seg) < window_size: return []
    wL = ref_seg.iloc[:window_size]
    if not window_is_finite(wL): return []
    vec_ref = window_vector(wL, L=window_size)
    blocks = enumerate_blocks(df, step_hours=step_hours, window_size=window_size)
    ex_margin = pd.Timedelta(days=ex_margin_days)
    F = len(FEAT_COLS); cand = []
    for b in blocks:
        if past_only:
            if not (b["end"] <= ref_range[0] - ex_margin): continue
        else:
            if not ((b["end"] <= ref_range[0] - ex_margin) or (b["start"] >= ref_range[1] + ex_margin)): continue
        w = df[(df["timestamp"] >= b["start"]) & (df["timestamp"] < b["end"])]
        if len(w) < window_size: continue
        wL2 = w.iloc[:window_size]
        if not window_is_finite(wL2): continue
        vec_hist = window_vector(wL2, L=window_size)
        sim = sim_tier3(vec_ref, vec_hist, L=window_size, F=F, mode=sim_engine, w_dtw=w_dtw)
        cand.append({"start": b["start"], "end": b["end"], "sim": sim})
    cand.sort(key=lambda x: x["sim"], reverse=True)
    return cand[:topN]

def _adjust_magnitude(pct_mag: float) -> float:
    if pct_mag < 0.5: return max(0.0, pct_mag - 0.1)
    elif pct_mag < 0.8: return max(0.0, pct_mag - 0.2)
    else: return max(0.0, pct_mag)

def _get_close_at_or_next(df: pd.DataFrame, ts: pd.Timestamp):
    row = df[df["timestamp"] == ts]
    if not row.empty: return float(row["close"].iloc[0])
    seg_after = df[df["timestamp"] > ts]
    if not seg_after.empty: return float(seg_after["close"].iloc[0])
    return None

def _touch_entry(df: pd.DataFrame, start_ts, end_ts, side: str, target_price: float):
    seg = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    if seg.empty: return None, None
    if side == "LONG":
        hit = seg[seg["low"] <= target_price]
        return (hit["timestamp"].iloc[0], float(target_price)) if not hit.empty else (None, None)
    else:
        hit = seg[seg["high"] >= target_price]
        return (hit["timestamp"].iloc[0], float(target_price)) if not hit.empty else (None, None)

def _risk_size_with_leverage(entry_price, sl, eq_run, risk_pct, max_leverage):
    if eq_run is None or eq_run <= 0: return 0.0, 0.0, False
    size_notional = float(eq_run) * float(max_leverage)
    used_lev = float(max_leverage)
    cap_hit = False
    return size_notional, used_lev, cap_hit

# ---------------------------
# NOW
# ---------------------------
if sim_mode == "NOW":
    st.subheader("NOW: 28h 지연 엔트리 · 1회 거래 (태그별 전략 명시 포함)")

    df_full = df_full_static  # NOW는 static 기준 사용

    # 후보 탐색
    cands = get_candidates(
        df_full, (ref_start, ref_end),
        ex_margin_days=10 if fast else 5,
        topN=5 if fast else 10,
        past_only=True
    )

    results = []
    stepTD = pd.Timedelta(hours=step_hours)
    for f in cands:
        next_start = f["end"]
        next_end = next_start + stepTD
        df_next = df_full[(df_full["timestamp"] >= next_start) & (df_full["timestamp"] < next_end)]
        if len(df_next) < window_size:
            continue
        closes = df_next["close"].to_numpy()
        base = float(df_next["open"].iloc[0])
        pct_raw = (closes - base) / base * 100.0
        # 안전하게 28h 종가(인덱스 L-1) 저장 (없으면 마지막 종가)
        base_close_28h = float(df_next["close"].iloc[window_size - 1]) if len(df_next) >= window_size else float(df_next["close"].iloc[-1])
        results.append({
            "sim": f["sim"],
            "next_start": next_start,
            "next_end": next_end,
            "pct": pct_raw,
            "df_next": df_next.reset_index(drop=True),
            "base_close": base,
            "base_close_28h": base_close_28h
        })

    # 엔트리 타이밍 체크
    t_entry = pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS)
    if now_ts < t_entry:
        st.info(f"데이터 부족: 엔트리 고려 시점({t_entry})까지 28h가 지나지 않음.")
        st.stop()

    cur_pred_seg = df_full[
        (df_full["timestamp"] >= pred_start) &
        (df_full["timestamp"] <= min(now_ts, pred_end))
    ]
    if len(cur_pred_seg) == 0 or len(results) == 0:
        st.info("데이터 부족")
        st.stop()

    base_cur = float(cur_pred_seg["open"].iloc[0])
    a_plot = ((cur_pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)

    prefix_end = min(t_entry, pred_end)
    cur_prefix = cur_pred_seg[cur_pred_seg["timestamp"] <= prefix_end]
    a = ((cur_prefix["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)
    L = len(a)

    # 프리픽스 최고 후보 선정
    best = None
    for r in results:
        b = np.array(r["pct"], dtype=float)[:L]
        sim_shape = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) else float(cosine_similarity([a], [b])[0][0])
        if (best is None) or (sim_shape > best["sim"]):
            best = {"sim": sim_shape, "flow": r}

    df_best_next = best["flow"]["df_next"]
    # 안전하게 base_hist_close 확보
    base_hist_close = best["flow"].get(
        "base_close_28h",
        best["flow"].get("base_close",
                         (float(df_best_next["close"].iloc[L-1]) if len(df_best_next) >= L and L>0 else float(df_best_next["close"].iloc[-1])))
    )

    # 표 — (원본 NOW는 expander였지만 유지)
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

    # 그래프
    fig, ax = plt.subplots(figsize=(9, 3))
    hist_full = np.array(best["flow"]["pct"], dtype=float)
    ax.plot(np.arange(len(hist_full)), hist_full, label="매칭 72h(원시%)")
    ax.plot(np.arange(len(a_plot)), a_plot, label=f"현재 진행 (L={len(a_plot)})")
    ax.axvline(L - 1, ls="--", label="엔트리 기준(28h)")
    ax.axhline(HI_THR, ls="--"); ax.axhline(-HI_THR, ls="--")
    ax.axhline(LO_THR, ls=":"); ax.axhline(-LO_THR, ls=":")
    ax.axhline(0, ls=":")
    ax.set_title("NOW: 28h 기준 · 진행 vs 매칭 (원시%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ---------------- NOW: 시나리오 비교 ----------------
    fut = hist_full[L - 1:] - (hist_full[L - 1] if L > 0 else 0.0)
    idx_max = int(np.argmax(fut)) if fut.size > 0 else 0
    idx_min = int(np.argmin(fut)) if fut.size > 0 else 0
    max_up = float(np.max(fut)) if fut.size > 0 else 0.0
    min_dn = float(np.min(fut)) if fut.size > 0 else 0.0

    ext_start = pred_start - pd.Timedelta(hours=48)
    prefix_end = min(t_entry, pred_end)
    ext_seg = df_full[
        (df_full["timestamp"] >= ext_start) &
        (df_full["timestamp"] <= prefix_end)
    ].reset_index(drop=True)
    used_ext = (len(ext_seg) >= 2)
    seg = ext_seg if len(ext_seg) >= 2 else cur_prefix

    anchor = float(seg["close"].iloc[0])
    last = float(seg["close"].iloc[-1])
    ret_pct = (last / anchor - 1.0) * 100.0
    thr_ext  = -1.0
    thr_cur  =  0.0
    cutoff   = (thr_ext if used_ext else thr_cur)
    regime_down = (ret_pct < cutoff)
    sim_gate = 0.75 + (0.05 if regime_down else 0.0)
    LO_THR_USE = LO_THR + (0.5 if regime_down else 0.0)
    HI_THR_USE = HI_THR + (0.5 if regime_down else 0.0)

    mag_up = abs(max_up); mag_dn = abs(min_dn); mag = max(mag_up, mag_dn)
    up_win = mag_up >= mag_dn; dn_win = mag_dn > mag_up

    if (mag >= HI_THR_USE) and up_win and (not regime_down):
        current_scenario = "A"
    elif (mag >= HI_THR_USE) and dn_win:
        current_scenario = "B"
    elif (LO_THR_USE <= mag < HI_THR_USE) and up_win and (not regime_down):
        current_scenario = "C"
    elif (LO_THR_USE <= mag < HI_THR_USE) and dn_win:
        current_scenario = "C′"
    elif (LO_THR_USE <= mag < HI_THR_USE) and up_win and regime_down:
        current_scenario = "D"
    else:
        current_scenario = "E"

    st.markdown(f"### 📌 현재 판정: **{current_scenario} 시나리오**")

    # 공용: 리밋 타겟 계산 (가정값)
    def compute_limit_target_local(side: str,
                                   df_next_best: pd.DataFrame,
                                   L_local: int,
                                   idx_max_local: int,
                                   idx_min_local: int,
                                   cur_28h_close_local: float,
                                   base_hist_close_28h_local: float):
        if side == "LONG":
            end_k = min((L_local - 1) + idx_max_local, len(df_next_best) - 1)
            lows_slice = df_next_best["low"].iloc[:end_k + 1].to_numpy(dtype=float)
            if lows_slice.size == 0:
                return None
            low_min = float(np.min(lows_slice))
            drop_pct = (low_min / base_hist_close_28h_local - 1.0) * 100.0
            mag_adj = _adjust_magnitude(abs(drop_pct))
            return cur_28h_close_local * (1.0 + (-mag_adj) / 100.0)
        elif side == "SHORT":
            end_k = min((L_local - 1) + idx_min_local, len(df_next_best) - 1)
            highs_slice = df_next_best["high"].iloc[:end_k + 1].to_numpy(dtype=float)
            if highs_slice.size == 0:
                return None
            high_max = float(np.max(highs_slice))
            up_pct = (high_max / base_hist_close_28h_local - 1.0) * 100.0
            mag_adj = _adjust_magnitude(abs(up_pct))
            return cur_28h_close_local * (1.0 + mag_adj / 100.0)
        return None

    def next_open_after_local(ts):
        seg_after = df_full[df_full["timestamp"] > ts]
        return (seg_after["timestamp"].iloc[0], float(seg_after["open"].iloc[0])) if not seg_after.empty else (None, None)

    force_mode = st.checkbox("강제 가정으로 계산(조건 불충족이어도 값 채움)", value=True)

    cur_28h_close = _get_close_at_or_next(df_full, t_entry)
    base_hist_close_local = base_hist_close

# ---------------------------
# LIVE (실거래)
# ---------------------------
elif sim_mode == "LIVE":
    st.subheader("LIVE: 실거래 (메인넷)")
    df_full = df_full_static

    entry_rule = ENTRY_RULE_FIXED
    sltp_method = "ATR"; k_sl, k_tp = 1.0, 3.0; sl_pct, tp_pct = -0.015, 0.03

    with st.expander("💳 계정 · 선물 지갑 (메인넷)", expanded=True):
        tclient = connect_binance_trade()
        trade_symbol = st.text_input("거래 심볼", value="BTCUSDT")
        leverage = st.number_input("레버리지(x)", 1, 100, 10, 1)
        margin_mode = st.radio("마진 모드", ["교차(Cross)", "격리(Isolated)"], index=0, horizontal=True)
        use_cross = (margin_mode == "교차(Cross)")
        size_pct = st.slider("사이즈 % (가용잔고 기준)", 0.1, 100.0, 2.0, 0.1)

        bals = get_futures_balances(tclient)
        colb1, colb2 = st.columns(2)
        colb1.metric("USDT Wallet", f"{bals['wallet_balance']:.2f}")
        colb2.metric("USDT Available", f"{bals['available_balance']:.2f}")

    (ref_start, ref_end), (pred_start, pred_end) = pick_blocks(df_full['timestamp'].iloc[-1], step_hours=step_hours)
    t_entry = pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS)

    entry_time, entry_price = make_entry_at(df_full, t_entry, rule=entry_rule)
    if entry_time is not None and entry_time < t_entry:
        seg_after = df_full[df_full["timestamp"] > t_entry]
        if not seg_after.empty:
            entry_time = seg_after["timestamp"].iloc[0]
            entry_price = float(seg_after["open"].iloc[0])

    atr_ref = float(df_full.loc[df_full["timestamp"] == entry_time, "atr"].fillna(method='ffill').iloc[0]) if entry_time is not None else None
    sl, tp = make_sl_tp(entry_price, "LONG", method=sltp_method, atr=atr_ref,
                        sl_pct=sl_pct, tp_pct=tp_pct, k_sl=k_sl, k_tp=k_tp, tick_size=0.0)

    tclient2 = connect_binance_trade()
    ensure_leverage_and_margin(tclient2, symbol=trade_symbol, leverage=int(leverage), cross=use_cross)
    tick_size, qty_step = get_symbol_filters(tclient2, symbol=trade_symbol)

    avail = bals["available_balance"]
    notional = max(0.0, avail) * float(leverage) * (float(size_pct) / 100.0)
    qty_live = (notional / float(entry_price)) if entry_price else 0.0

    st.markdown("### 📌 주문 미리보기")
    colp1, colp2, colp3, colp4 = st.columns(4)
    colp1.metric("Entry(참조)", f"{(entry_price or 0):.2f}")
    colp2.metric("SL", f"{(sl or 0):.2f}")
    colp3.metric("TP", f"{(tp or 0):.2f}")
    colp4.metric("수량(계약)", f"{qty_live:.6f}")

# ---------------------------
# ROLLING (원본)
# ---------------------------
elif sim_mode == "ROLLING":
    st.subheader("ROLLING: 28h 지연 엔트리 · 블록당 1회 거래 백테스트 (Static only)")

    # 공통 파라미터
    topN = 5 if fast else 10
    exd = 10 if fast else 5
    stepTD = pd.Timedelta(hours=step_hours)
    delayTD = pd.Timedelta(hours=ENTRY_DELAY_HOURS)

    # 백테스트 예측 구간 시작(현재 분석 구간)
    ROLL_START = pd.Timestamp("2025-01-01 00:00:00")

    # 블록 시퀀스 기준(공통) — static으로 시간축 고정
    df_roll_base = df_full_static[df_full_static["timestamp"] >= (ROLL_START - pd.Timedelta(hours=72))].reset_index(drop=True)
    if len(df_roll_base) < window_size:
        st.warning("ROLLING: 데이터 부족")
        st.stop()

    blocks_all = enumerate_blocks(df_roll_base, step_hours=step_hours, window_size=window_size)
    start_idx = None
    for i in range(1, len(blocks_all)):
        if blocks_all[i]["start"] >= ROLL_START:
            start_idx = i
            break
    if start_idx is None:
        st.warning("ROLLING: 2025년 이후 pred 블록 없음")
        st.stop()

    # 후보 기간 정책 (static만 사용): 2025-01-01 이후
    hist_start_static = pd.Timestamp("2025-01-01 00:00:00")

    # 단일 variant 평가 함수
    def _eval_variant(df_full_var, ref_b, pred_b, hist_start):
        # 예측 프리픽스는 공통적으로 '현재(2025+)' 사용(시간축 동기)
        df_roll = df_full_static[df_full_static["timestamp"] >= (ROLL_START - pd.Timedelta(hours=72))].reset_index(drop=True)

        # 후보 풀
        df_hist = df_full_var[df_full_var["timestamp"] >= hist_start].reset_index(drop=True)

        # 후보
        cands = get_candidates(df_hist, (ref_b["start"], ref_b["end"]), ex_margin_days=exd, topN=topN, past_only=True)
        if not cands:
            return None

        # 후보의 다음 72h (종가 기반)
        results = []
        for f in cands:
            next_start = f["end"]
            next_end = next_start + stepTD
            df_next = df_hist[(df_hist["timestamp"] >= next_start) & (df_hist["timestamp"] < next_end)]
            if len(df_next) < window_size:
                continue
            closes = df_next["close"].to_numpy()
            baseC = float(closes[0])
            pct_c = (closes - baseC) / baseC * 100.0
            results.append({
                "sim": f["sim"], "next_start": next_start, "next_end": next_end,
                "pct": pct_c, "df_next": df_next.reset_index(drop=True), "base_close": baseC
            })

        if not results:
            return None

        # 현재 pred 프리픽스(종가 기준)
        t_entry = pred_b["start"] + delayTD
        if t_entry > pred_b["end"]:
            return None

        pred_seg = df_roll[(df_roll["timestamp"] >= pred_b["start"]) & (df_roll["timestamp"] <= t_entry)]
        if len(pred_seg) == 0:
            return None

        base_cur = float(pred_seg["close"].iloc[0])
        a = ((pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)
        L = len(a)

        # 프리픽스 유사도 최고 후보
        best = None
        for r in results:
            b = np.array(r["pct"], dtype=float)[:L]
            sim_shape = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) else float(cosine_similarity([a], [b])[0][0])
            if (best is None) or (sim_shape > best["sim"]):
                best = {"sim": sim_shape, "flow": r}

        return {"df_roll": df_roll, "df_hist": df_hist, "best": best, "L": L, "t_entry": t_entry, "pred_seg": pred_seg}

    trade_logs = []
    pbar = st.progress(0)
    total = (len(blocks_all) - start_idx)
    eq_run = float(equity)  # ▶ 트레이드별 자본 추적

    for j, bp in enumerate(range(start_idx, len(blocks_all))):
        ref_b = blocks_all[bp - 1]
        pred_b = blocks_all[bp]

        res_static = _eval_variant(
            df_full_static, ref_b, pred_b,
            hist_start=hist_start_static
        )

        if res_static is None:
            pbar.progress(int(100 * (j + 1) / total))
            continue

        # 단일 선택
        choose = res_static
        choose_scaler = "static"

        df_roll = choose["df_roll"]
        df_hist = choose["df_hist"]
        best = choose["best"]
        L = choose["L"]
        t_entry = choose["t_entry"]
        pred_seg = choose["pred_seg"]

        hist_full = np.array(best["flow"]["pct"], dtype=float)
        base_now = float(hist_full[L - 1])  # 후보 28h 시점 종가 변화율
        fut = hist_full[L - 1:] - base_now  # 이후 구간 변화율(후행)

        # 방향/크기
        idx_max = int(np.argmax(fut))
        idx_min = int(np.argmin(fut))
        max_up = float(np.max(fut))   # 양수
        min_dn = float(np.min(fut))   # 음수

        ext_start = pred_b["start"] - pd.Timedelta(hours=48)
        prefix_end = min(t_entry, pred_b["end"])
        ext_seg = df_roll[
            (df_roll["timestamp"] >= ext_start) &
            (df_roll["timestamp"] <= prefix_end)
        ].reset_index(drop=True)
        used_ext = (len(ext_seg) >= 2)
        seg = ext_seg if used_ext else pred_seg  # pred_seg은 기존 프리픽스 0~28h
       
        anchor = float(seg["close"].iloc[0])
        last   = float(seg["close"].iloc[-1])
        ret_pct = (last / anchor - 1.0) * 100.0

        thr_ext  = -1.0
        thr_pred =  0.0
        cutoff   = (thr_ext if used_ext else thr_pred)
        regime_down = (ret_pct < cutoff)
        sim_gate = 0.75 + (0.05 if regime_down else 0.0)
        LO_THR_USE = LO_THR + (0.5 if regime_down else 0.0)
        HI_THR_USE = HI_THR + (0.5 if regime_down else 0.0)

        side = "HOLD"
        skip_reason = None
        if best["sim"] >= sim_gate:
            mag_up = abs(max_up)
            mag_dn = abs(min_dn)
            mag = max(mag_up, mag_dn)
            if mag >= LO_THR_USE:
                if regime_down and (mag_up >= mag_dn):
                    side = "HOLD"
                else:
                    side = "LONG" if mag_up >= mag_dn else "SHORT"

        # 전략 분기
        entry_time, entry_price, entry_target = (None, None, None)

        if side in ("LONG", "SHORT"):
            if max(abs(max_up), abs(min_dn)) >= HI_THR_USE:
                # ---- 기존 전략: t_entry에서 즉시 진입 ----
                etime, eprice = make_entry_at(df_roll, t_entry, rule=ENTRY_RULE_FIXED)
                if etime is not None and etime < t_entry:
                    seg_after = df_roll[df_roll["timestamp"] > t_entry]
                    if not seg_after.empty:
                        etime = seg_after["timestamp"].iloc[0]
                        eprice = float(seg_after["open"].iloc[0])
                entry_time, entry_price = etime, eprice
            else:
                # ---- 새 전략: 되돌림 매핑 리밋 + 터치체결 ----
                df_next_best = best["flow"]["df_next"]
                base_hist_close = float(best["flow"]["base_close"])  # 후보 28h 종가
                cur_28h_close = _get_close_at_or_next(df_roll, t_entry)

                if cur_28h_close is not None:
                    if side == "LONG":
                        end_k = min((L - 1) + idx_max, len(df_next_best) - 1)
                        lows_slice = df_next_best["low"].iloc[:end_k + 1].to_numpy(dtype=float)
                        if lows_slice.size > 0:
                            low_min = float(np.min(lows_slice))
                            drop_pct = (low_min / base_hist_close - 1.0) * 100.0  # 음수
                            mag_adj = _adjust_magnitude(abs(drop_pct))
                            drop_adj = -mag_adj
                            entry_target = cur_28h_close * (1.0 + drop_adj / 100.0)
                            entry_time, entry_price = _touch_entry(df_roll, t_entry, pred_b["end"], "LONG", entry_target)
                    else:
                        end_k = min((L - 1) + idx_min, len(df_next_best) - 1)
                        highs_slice = df_next_best["high"].iloc[:end_k + 1].to_numpy(dtype=float)
                        if highs_slice.size > 0:
                            high_max = float(np.max(highs_slice))
                            up_pct = (high_max / base_hist_close - 1.0) * 100.0  # 양수
                            mag_adj = _adjust_magnitude(abs(up_pct))
                            up_adj = mag_adj
                            entry_target = cur_28h_close * (1.0 + up_adj / 100.0)
                            entry_time, entry_price = _touch_entry(df_roll, t_entry, pred_b["end"], "SHORT", entry_target)

        # SL/TP & 사이징 & 시뮬
        atr_ref = None
        if entry_time is not None:
            row_at = df_roll[df_roll["timestamp"] == entry_time]
            if not row_at.empty and row_at["atr"].notna().any():
                atr_ref = float(row_at["atr"].fillna(method='ffill').iloc[0])

        sl, tp = (None, None)
        if side in ("LONG", "SHORT") and entry_time is not None and entry_price is not None:
            sl, tp = make_sl_tp(
                entry_price, side, method=sltp_method, atr=atr_ref,
                sl_pct=sl_pct, tp_pct=tp_pct, k_sl=k_sl, k_tp=k_tp, tick_size=0.0
            )

        size = 0.0
        used_lev = 0.0
        cap_hit = False
        if side in ("LONG", "SHORT") and entry_time is not None and entry_price is not None and sl:
            size, used_lev, cap_hit = _risk_size_with_leverage(entry_price, sl, eq_run, risk_pct, max_leverage)

        exit_time, exit_price, gross_ret, net_ret = (None, None, None, None)
        if side in ("LONG", "SHORT") and entry_time is not None and entry_price is not None:
            exit_time, exit_price, gross_ret, net_ret = simulate_trade(
                df_roll, t_entry, pred_b["end"], side,
                entry_time, entry_price, sl, tp,
                fee_entry=fee_entry, fee_exit=fee_exit,
                slip_entry=slip_entry, slip_exit=slip_exit
            )
        else:
            if side in ("LONG", "SHORT"):
                side = "HOLD"  # 미터치·미진입

        # ▶ 자본 기반 수치 업데이트
        ret_pct = (net_ret or 0.0) / 100.0
        eq_before = eq_run
        pnl_usd = (size or 0.0) * ret_pct
        eq_run = eq_run + pnl_usd
        ret_equity_pct = (pnl_usd / (eq_before if eq_before > 0 else 1.0)) * 100.0

        trade_logs.append({
            "pred_start": pred_b["start"],
            "pred_end": pred_b["end"],
            "t_entry": t_entry,
            "side": side,
            "sim_prefix": best["sim"],
            "scaler": choose_scaler,
            "entry_time": entry_time,
            "entry": entry_price,
            "entry_target": entry_target,
            "SL": sl,
            "TP": tp,
            "size_notional": size,
            "used_lev": used_lev,
            "cap_hit": cap_hit,
            "exit_time": exit_time,
            "exit": exit_price,
            "gross_ret_%": gross_ret,
            "net_ret_%": net_ret,
            "eq_before": eq_before,
            "eq_after": eq_run,
            "pnl_usd": pnl_usd,
            "ret_equity_%": ret_equity_pct,
            "skip_reason": skip_reason,
        })

        pbar.progress(int(100 * (j + 1) / total))

    if not trade_logs:
        st.info("ROLLING 결과 없음")
        st.stop()

    df_log = pd.DataFrame(trade_logs).sort_values("pred_start").reset_index(drop=True)

    # ===== 표시용 테이블: 가격기준 % 숨기고 레버리지 반영 %를 메인으로 =====
    df_show = df_log.copy()
    df_show = df_show.drop(columns=["gross_ret_%", "net_ret_%"], errors="ignore")
    df_show = df_show.rename(columns={"ret_equity_%": "ret_%(levered)"})
    cols = [
        "pred_start", "pred_end", "t_entry", "side", "sim_prefix", "scaler",
        "entry_time", "entry", "entry_target", "SL", "TP",
        "size_notional", "used_lev", "cap_hit", "pnl_usd", "ret_%(levered)",
        "eq_before", "eq_after", "exit_time", "exit", "skip_reason"
    ]
    df_show = df_show[[c for c in cols if c in df_show.columns]]

    st.markdown("### 결과 테이블 (레버리지 반영 수익률)")
    st.caption("ret_%(levered) = net_ret_% × (size_notional / eq_before)")
    st.dataframe(df_show, use_container_width=True)

    # -------------------------------
    # 에쿼티 커브 & 카드
    # -------------------------------
    if 'df_log' in locals() and df_log is not None and not df_log.empty:
        dates, equity_curve = build_equity_curve(df_log, float(equity))
        metrics = calc_metrics(df_log, equity_curve)

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

# ---------------------------
# NOW (A) — 원본 NOW와 동일한 UI/표/그래프/계산 흐름 + expander 없이 표 즉시 표시
# ---------------------------
elif sim_mode == "NOW (A)":
    st.subheader("NOW (A): 28h 지연 엔트리 · 1회 거래 (NOW 모드와 동일 UI/동작)")

    # A-logic 전용 POOL 설정
    Z_TRAIN_END = pd.Timestamp("2021-07-01 00:00:00")
    POOL_START  = pd.Timestamp("2021-04-01 00:00:00")
    POOL_END    = pd.Timestamp("2021-07-01 00:00:00")

    # 데이터 준비
    df_full_a = apply_static_zscore(df_feat.copy(), GLOBAL_Z_COLS, Z_TRAIN_END)
    df_full_a = finalize_preprocessed(df_full_a, window_size)
    pool_df   = df_full_a[
        (df_full_a["timestamp"] >= POOL_START) &
        (df_full_a["timestamp"] < POOL_END)
    ].reset_index(drop=True)

    if len(pool_df) < window_size:
        st.error("POOL 구간 데이터가 부족합니다.")
        st.stop()

    # 블록/엔트리
    (ref_start, ref_end), (pred_start, pred_end) = pick_blocks(
        df_full_a['timestamp'].iloc[-1], step_hours=step_hours
    )
    t_entry = pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS)

    # 현재시간 기준 28h 경과 체크
    if now_ts < t_entry:
        st.info(f"데이터 부족: 엔트리 고려 시점({t_entry})까지 28h가 지나지 않음.")
        st.stop()

    # 후보 탐색
    cands_a = get_candidates_a(
        pool_df, (ref_start, ref_end),
        df_full_a, FEAT_COLS,
        step_hours=step_hours, window_size=window_size,
        sim_mode="DTW", w_dtw=0.5, topN=20, ex_margin_days=5
    )
    if not cands_a:
        st.info("후보 없음 (POOL 협소)")
        st.stop()

    # 후보 다음 72h
    results = []
    stepTD = pd.Timedelta(hours=step_hours)
    for f in cands_a:
        next_start = f["end"]
        next_end   = next_start + stepTD
        df_next = pool_df[
            (pool_df["timestamp"] >= next_start) &
            (pool_df["timestamp"] <  next_end)
        ]
        if len(df_next) < window_size:
            continue
        if len(df_next) < ENTRY_DELAY_HOURS:  # 28h 없는 후보 배제
            continue

        closes = df_next["close"].to_numpy()
        baseC  = float(df_next["open"].iloc[0])
        pct_c  = (closes - baseC) / baseC * 100.0
        base_close_28h = float(df_next["close"].iloc[ENTRY_DELAY_HOURS - 1])

        results.append({
            "sim": f.get("sim", np.nan),
            "next_start": next_start,
            "next_end":   next_end,
            "pct": pct_c,
            "df_next": df_next.reset_index(drop=True),
            "base_close": baseC,
            "base_close_28h": base_close_28h
        })

    if not results:
        st.info("후보 결과 부족 (28h 지점 없음)")
        st.stop()

    # 현재 프리픽스
    cur_pred_seg = df_full_a[
        (df_full_a["timestamp"] >= pred_start) &
        (df_full_a["timestamp"] <= min(now_ts, pred_end))
    ]
    if len(cur_pred_seg) == 0:
        st.info("프리픽스 데이터 없음")
        st.stop()

    base_cur = float(cur_pred_seg["open"].iloc[0])
    a = ((cur_pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)
    L = len(a)

    # ✅ cur_prefix 정의 (NOW와 동일)
    prefix_end = min(t_entry, pred_end)
    cur_prefix = cur_pred_seg[cur_pred_seg["timestamp"] <= prefix_end]

    # 프리픽스-형상 유사도 최고
    best = None
    for r in results:
        b = np.array(r["pct"], dtype=float)[:L]
        sim_shape = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) \
                    else float(cosine_similarity([a], [b])[0][0])
        if (best is None) or (sim_shape > best["sim"]):
            best = {"sim": sim_shape, "flow": r}

    if best is None:
        st.info("매칭 실패")
        st.stop()

    hist_full = np.array(best["flow"]["pct"], dtype=float)
    df_best_next = best["flow"]["df_next"]

    # 진행 그래프용 (NOW와 동일 → cur_pred_seg 사용)
    a_plot = ((cur_pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)

    # 표 (expander)
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

    # 시간 정보
    st.markdown("### ⏱️ 시간 정보")
    st.write({
        "현재 블록 구간": f"{pred_start} ~ {pred_end}",
        "28h 지연 엔트리 시점": str(t_entry)
    })

    # 그래프
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(np.arange(len(hist_full)), hist_full, label="매칭 72h(원시%)")
    ax.plot(np.arange(len(a_plot)), a_plot, label=f"현재 진행 (L={len(a_plot)})")
    ax.axvline(L - 1, ls="--", label="엔트리 기준(28h)")
    ax.axhline(HI_THR, ls="--"); ax.axhline(-HI_THR, ls="--")
    ax.axhline(LO_THR, ls=":"); ax.axhline(-LO_THR, ls=":")
    ax.axhline(0, ls=":")
    ax.set_title("NOW (A): 28h 기준 · 진행 vs 매칭 (원시%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # 강제 계산 스위치
    force_mode = st.checkbox("강제 가정으로 계산(조건 불충족이어도 값 채움)", value=True)

    # ---------------- 리밋 타겟 계산 함수 ----------------
    def compute_limit_target_local(side: str,
                                   df_next_best: pd.DataFrame,
                                   L_local: int,
                                   idx_max_local: int,
                                   idx_min_local: int,
                                   cur_28h_close_local: float,
                                   base_hist_close_28h_local: float):
        if side == "LONG":
            end_k = min((L_local - 1) + idx_max_local, len(df_next_best) - 1)
            lows_slice = df_next_best["low"].iloc[:end_k + 1].to_numpy(dtype=float)
            if lows_slice.size == 0:
                return None
            low_min = float(np.min(lows_slice))
            drop_pct = (low_min / base_hist_close_28h_local - 1.0) * 100.0
            mag_adj = _adjust_magnitude(abs(drop_pct))
            return cur_28h_close_local * (1.0 + (-mag_adj) / 100.0)

        elif side == "SHORT":
            end_k = min((L_local - 1) + idx_min_local, len(df_next_best) - 1)
            highs_slice = df_next_best["high"].iloc[:end_k + 1].to_numpy(dtype=float)
            if highs_slice.size == 0:
                return None
            high_max = float(np.max(highs_slice))
            up_pct = (high_max / base_hist_close_28h_local - 1.0) * 100.0
            mag_adj = _adjust_magnitude(abs(up_pct))
            return cur_28h_close_local * (1.0 + mag_adj / 100.0)

        return None

    # ---------------- 시나리오 계산 ----------------
    fut = hist_full[L - 1:] - (hist_full[L - 1] if L > 0 else 0.0)
    idx_max = int(np.argmax(fut)) if fut.size > 0 else 0
    idx_min = int(np.argmin(fut)) if fut.size > 0 else 0
    max_up  = float(np.max(fut)) if fut.size > 0 else 0.0
    min_dn  = float(np.min(fut)) if fut.size > 0 else 0.0

    ext_start = pred_start - pd.Timedelta(hours=48)
    prefix_end = min(t_entry, pred_end)
    ext_seg = df_full_a[
        (df_full_a["timestamp"] >= ext_start) &
        (df_full_a["timestamp"] <= prefix_end)
    ].reset_index(drop=True)
    used_ext = (len(ext_seg) >= 2)
    seg = ext_seg if used_ext else cur_prefix

    anchor = float(seg["close"].iloc[0])
    last   = float(seg["close"].iloc[-1])
    ret_pct = (last / anchor - 1.0) * 100.0
    cutoff  = (-1.0 if used_ext else 0.0)
    regime_down = (ret_pct < cutoff)

    sim_gate = 0.9 + (0.05 if regime_down else 0.0)
    LO_THR_USE = LO_THR + (0.5 if regime_down else 0.0)
    HI_THR_USE = HI_THR + (0.5 if regime_down else 0.0)

    up_win = (abs(max_up) >= abs(min_dn))
    dn_win = not up_win
    mag = max(abs(max_up), abs(min_dn))

    if (mag >= HI_THR_USE) and up_win and not regime_down:
        scenario = "A"
    elif (mag >= HI_THR_USE) and dn_win:
        scenario = "B"
    elif (LO_THR_USE <= mag < HI_THR_USE) and up_win and not regime_down:
        scenario = "C"
    elif (LO_THR_USE <= mag < HI_THR_USE) and dn_win:
        scenario = "C′"
    elif (LO_THR_USE <= mag < HI_THR_USE) and up_win and regime_down:
        scenario = "D"
    else:
        scenario = "E"

    st.markdown(f"### 📌 현재 판정(A-logic): **{scenario}**")

    # ---------------- 진입/익절/손절가 출력 ----------------
    entry_price = _get_close_at_or_next(df_full_static, t_entry)  # 원본에서 확보
    base_hist_close_local = best["flow"]["base_close_28h"]

    long_target  = compute_limit_target_local("LONG",  df_best_next, L, idx_max, idx_min, entry_price, base_hist_close_local)
    short_target = compute_limit_target_local("SHORT", df_best_next, L, idx_max, idx_min, entry_price, base_hist_close_local)

    side, tp_price, sl_price = "NO TRADE", None, None
    if scenario in ["A", "C", "D"]:   # 상승 계열
        side = "LONG"
        tp_price = long_target
        sl_price = short_target
    elif scenario in ["B", "C′"]:     # 하락 계열
        side = "SHORT"
        tp_price = short_target
        sl_price = long_target

    st.markdown("### 🎯 포지션 정보")
    st.write({
        "시나리오": scenario,
        "진입 방향": side,
        "진입가 (Entry)": entry_price,
        "익절가 (Take Profit)": tp_price,
        "손절가 (Stop Loss)": sl_price
    })

# ---------------------------
# ROLLING (A) — expander/버튼 제거, 즉시 실행 · 원본 ROLLING 레이아웃/지표 동일
# ---------------------------
elif sim_mode == "ROLLING (A)":
    st.subheader("ROLLING (A): 제한 POOL + 원본 레이아웃/지표 — 즉시 실행")

    # 고정 POOL
    Z_TRAIN_END = pd.Timestamp("2021-07-01 00:00:00")
    POOL_START  = pd.Timestamp("2021-04-01 00:00:00")
    POOL_END    = pd.Timestamp("2021-07-01 00:00:00")

    # 파라미터 (expander/버튼 없이 바로)
    col1, col2, col3 = st.columns(3)
    with col1:
        LO_THR = st.number_input("LO_THR (%)", 0.0, 10.0, 1.0, 0.1)
        HI_THR = st.number_input("HI_THR (%)", 0.0, 20.0, 3.0, 0.1)
        sim_engine_a = st.selectbox("Similarity(A)", ["DTW","Frechet","Hybrid","Cosine"], index=0)
        w_dtw_a = st.slider("Hybrid: DTW weight (A)", 0.0, 1.0, 0.5, 0.05)
    with col2:
        sim_gate_base = st.slider("Similarity gate (base)", 0.0, 1.0, 0.75, 0.01)
        topN = st.slider("Candidates topN", 5, 50, 5, 1)
        ex_margin_days = st.slider("Exclude margin (days)", 0, 30, 5, 1)
        ROLL_START = pd.Timestamp("2025-01-01 00:00:00")
        # datetime -> Timestamp 보정
        if not isinstance(ROLL_START, pd.Timestamp):
            ROLL_START = pd.Timestamp(ROLL_START)
    with col3:
        sltp_method_a = st.radio("SL/TP", ["ATR","FIXED%"], index=0, horizontal=True)
        if sltp_method_a == "ATR":
            k_sl_a = st.number_input("k_sl (×ATR)", 0.1, 10.0, 1.5, 0.1)
            k_tp_a = st.number_input("k_tp (×ATR)", 0.1, 15.0, 3.0, 0.1)
            sl_pct_a, tp_pct_a = -0.015, 0.03
        else:
            sl_pct_a = st.number_input("SL % (음수)", -50.0, 0.0, -1.5, 0.1) / 100.0
            tp_pct_a = st.number_input("TP %", 0.0, 100.0, 3.0, 0.1) / 100.0
            k_sl_a, k_tp_a = 1.0, 2.0
        fee_entry = st.number_input("Entry fee %", 0.0, 1.0, 0.04, 0.01) / 100.0
        fee_exit  = st.number_input("Exit fee %", 0.0, 1.0, 0.05, 0.01) / 100.0
        slip_entry = st.number_input("Slippage entry %", 0.0, 1.0, 0.03, 0.01) / 100.0
        slip_exit  = st.number_input("Slippage exit %", 0.0, 1.0, 0.05, 0.01) / 100.0

    # 데이터/POOL 준비
    df_full_a = apply_static_zscore(df_feat.copy(), GLOBAL_Z_COLS, Z_TRAIN_END)
    df_full_a = finalize_preprocessed(df_full_a, window_size)
    pool_df   = df_full_a[(df_full_a["timestamp"] >= POOL_START) & (df_full_a["timestamp"] < POOL_END)].reset_index(drop=True)

    # 블록 나누기 (원본과 동일)
    df_roll_base = df_full_a[df_full_a["timestamp"] >= (ROLL_START - pd.Timedelta(hours=72))].reset_index(drop=True)
    blocks_all = enumerate_blocks(df_roll_base, step_hours=step_hours, window_size=window_size)

    start_idx = None
    for i in range(1, len(blocks_all)):
        if blocks_all[i]["start"] >= ROLL_START:
            start_idx = i
            break
    if start_idx is None:
        st.warning("ROLLING(A): 지정 시작 이후 pred 블록 없음")
        st.stop()

    # 즉시 실행
    trade_logs = []
    stepTD = pd.Timedelta(hours=step_hours)
    delayTD = pd.Timedelta(hours=ENTRY_DELAY_HOURS)
    pbar = st.progress(0); total = (len(blocks_all) - start_idx)
    eq_run = float(equity)

    for j, bp in enumerate(range(start_idx, len(blocks_all))):
        ref_b = blocks_all[bp - 1]; pred_b = blocks_all[bp]
        t_entry = pred_b["start"] + delayTD
        if t_entry > pred_b["end"]:
            pbar.progress(int(100*(j+1)/max(1,total)))
            continue

        cands = get_candidates_a(
            pool_df, (ref_b["start"], ref_b["end"]), df_full_a, FEAT_COLS,
            step_hours=step_hours, window_size=window_size,
            sim_mode=sim_engine_a, w_dtw=w_dtw_a, topN=topN, ex_margin_days=ex_margin_days
        )
        if not cands:
            pbar.progress(int(100*(j+1)/max(1,total)))
            continue

        results = []
        for f in cands:
            next_start = f["end"]; next_end = next_start + stepTD
            df_next = pool_df[(pool_df["timestamp"] >= next_start) & (pool_df["timestamp"] < next_end)]
            if len(df_next) < window_size:
                continue
            closes = df_next["close"].to_numpy(); baseC = float(closes[0])
            pct_c  = (closes - baseC) / baseC * 100.0
            results.append({
                "sim": f["sim"], "next_start": next_start, "next_end": next_end,
                "pct": pct_c, "df_next": df_next.reset_index(drop=True), "base_close": baseC
            })
        if not results:
            pbar.progress(int(100*(j+1)/max(1,total)))
            continue

        pred_seg = df_full_a[(df_full_a["timestamp"] >= pred_b["start"]) & (df_full_a["timestamp"] <= t_entry)]
        if len(pred_seg) == 0:
            pbar.progress(int(100*(j+1)/max(1,total)))
            continue

        base_cur = float(pred_seg["close"].iloc[0])
        a = ((pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float); L = len(a)

        best = None
        for r in results:
            b = np.array(r["pct"], dtype=float)[:L]
            sim_shape = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) else float(cosine_similarity([a], [b])[0][0])
            if (best is None) or (sim_shape > best["sim"]):
                best = {"sim": sim_shape, "flow": r}

        hist_full = np.array(best["flow"]["pct"], dtype=float)
        base_now  = float(hist_full[L - 1]); fut = hist_full[L - 1:] - base_now
        idx_max   = int(np.argmax(fut)); idx_min = int(np.argmin(fut))
        max_up    = float(np.max(fut));  min_dn  = float(np.min(fut))

        ext_start = pred_b["start"] - pd.Timedelta(hours=48)
        prefix_end = min(t_entry, pred_b["end"])
        ext_seg = df_full_a[(df_full_a["timestamp"] >= ext_start) & (df_full_a["timestamp"] <= prefix_end)].reset_index(drop=True)
        used_ext = (len(ext_seg) >= 2)
        seg = ext_seg if used_ext else pred_seg
        anchor = float(seg["close"].iloc[0]); last = float(seg["close"].iloc[-1])
        ret_pct = (last / anchor - 1.0) * 100.0

        # === 결정/주문/시뮬 전 구간 (추가) ===
        cutoff  = (-1.0 if used_ext else 0.0)
        regime_down = (ret_pct < cutoff)
        sim_gate = float(sim_gate_base) + (0.05 if regime_down else 0.0)
        LO_THR_USE = LO_THR + (0.5 if regime_down else 0.0)
        HI_THR_USE = HI_THR + (0.5 if regime_down else 0.0)

        # 진입 방향 결정
        side = "HOLD"
        if best["sim"] >= sim_gate:
            mag_up = abs(max_up); mag_dn = abs(min_dn); mag = max(mag_up, mag_dn)
            if mag >= LO_THR_USE:
                if regime_down and (mag_up >= mag_dn):
                    side = "HOLD"
                else:
                    side = "LONG" if mag_up >= mag_dn else "SHORT"

        # 진입 가격/시간 계산
        entry_time = entry_price = entry_target = None
        if side in ("LONG","SHORT"):
            if max(abs(max_up), abs(min_dn)) >= HI_THR_USE:
                # HI 이상: t_entry에서 즉시 진입(다음봉 시가)
                et0, ep0 = make_entry_at(df_full_a, t_entry, rule="다음봉 시가")
                seg_after = df_full_a[df_full_a["timestamp"] > t_entry]
                if et0 is not None and et0 < t_entry and not seg_after.empty:
                    et0, ep0 = seg_after["timestamp"].iloc[0], float(seg_after["open"].iloc[0])
                entry_time, entry_price = et0, ep0
            else:
                # LO~HI: 되돌림 리밋 매핑 → 터치체결
                df_next_best = best["flow"]["df_next"]
                base_hist_close = float(best["flow"]["base_close"])
                cur_28h_close = _get_close_at_or_next_a(df_full_a, t_entry)
                if (cur_28h_close is not None) and (len(df_next_best) > 0):
                    if side == "LONG":
                        end_k = min((L - 1) + idx_max, len(df_next_best) - 1)
                        lows_slice = df_next_best["low"].iloc[:end_k + 1].to_numpy(dtype=float)
                        if lows_slice.size > 0:
                            low_min = float(np.min(lows_slice))
                            drop_pct = (low_min / base_hist_close - 1.0) * 100.0
                            mag_adj = _adjust_magnitude_a(abs(drop_pct))
                            entry_target = cur_28h_close * (1.0 - mag_adj/100.0)
                            entry_time, entry_price = _touch_entry_a(df_full_a, t_entry, pred_b["end"], "LONG", entry_target)
                    else:
                        end_k = min((L - 1) + idx_min, len(df_next_best) - 1)
                        highs_slice = df_next_best["high"].iloc[:end_k + 1].to_numpy(dtype=float)
                        if highs_slice.size > 0:
                            high_max = float(np.max(highs_slice))
                            up_pct = (high_max / base_hist_close - 1.0) * 100.0
                            mag_adj = _adjust_magnitude_a(abs(up_pct))
                            entry_target = cur_28h_close * (1.0 + mag_adj/100.0)
                            entry_time, entry_price = _touch_entry_a(df_full_a, t_entry, pred_b["end"], "SHORT", entry_target)

        # SL/TP 계산
        atr_ref = None
        if entry_time is not None:
            row_at = df_full_a[df_full_a["timestamp"] == entry_time]
            if not row_at.empty and row_at["atr"].notna().any():
                atr_ref = float(row_at["atr"].fillna(method='ffill').iloc[0])

        SL = TP = None
        if (side in ("LONG","SHORT")) and (entry_time is not None) and (entry_price is not None):
            SL, TP = make_sl_tp(
                entry_price, side,
                method=("ATR" if sltp_method_a=="ATR" else "FIXED"),
                atr=atr_ref, sl_pct=sl_pct_a, tp_pct=tp_pct_a,
                k_sl=k_sl_a, k_tp=k_tp_a, tick_size=0.0
            )
        else:
            side = "HOLD"

        # 사이징 & 체결 시뮬
        size = used_lev = 0.0; cap_hit = False
        exit_time = exit_price = gross_ret = net_ret = None
        if side in ("LONG","SHORT") and (entry_time is not None) and (entry_price is not None) and (SL is not None):
            size, used_lev, cap_hit = _risk_size_with_leverage(entry_price, SL, eq_run, risk_pct, max_leverage)
            exit_time, exit_price, gross_ret, net_ret = simulate_trade(
                df=df_full_a, start_ts=pred_b["start"], end_ts=pred_b["end"],
                side=side, entry_time=entry_time, entry_price=float(entry_price),
                sl=SL, tp=TP,
                fee_entry=fee_entry, fee_exit=fee_exit,
                slip_entry=slip_entry, slip_exit=slip_exit,
                exit_on_close=True
            )
        else:
            side = "HOLD"

        # 자본 업데이트
        ret_pct_trade = (net_ret or 0.0) / 100.0
        eq_before = eq_run
        pnl_usd = (size or 0.0) * ret_pct_trade
        eq_run = eq_run + pnl_usd
        ret_equity_pct = (pnl_usd / (eq_before if eq_before > 0 else 1.0)) * 100.0

        # 로그 적재
        trade_logs.append({
            "pred_start": pred_b["start"], "pred_end": pred_b["end"],
            "t_entry": t_entry, "side": side,
            "entry_time": entry_time, "entry": entry_price, "entry_target": entry_target,
            "SL": SL, "TP": TP, "exit_time": exit_time, "exit": exit_price,
            "gross_ret_%": gross_ret, "net_ret_%": net_ret,
            "size_notional": size, "used_lev": used_lev, "cap_hit": cap_hit,
            "eq_before": eq_before, "eq_after": eq_run, "pnl_usd": pnl_usd, "ret_equity_%": ret_equity_pct,
            "sim_shape": best["sim"]
        })

        pbar.progress(int(100*(j+1)/max(1,total)))

    # 결과 출력
    if not trade_logs:
        st.info("ROLLING(A) 결과 없음"); st.stop()

    df_log = pd.DataFrame(trade_logs).sort_values("pred_start").reset_index(drop=True)

    # 표시 테이블
    df_show = (df_log.copy()
               .drop(columns=["gross_ret_%","net_ret_%"], errors="ignore")
               .rename(columns={"ret_equity_%": "ret_%(levered)"}))
    cols = ["pred_start","pred_end","t_entry","side","entry_time","entry","entry_target",
            "SL","TP","size_notional","used_lev","cap_hit","pnl_usd","ret_%(levered)",
            "eq_before","eq_after","exit_time","exit","sim_shape"]
    df_show = df_show[[c for c in cols if c in df_show.columns]]

    st.markdown("### 결과 테이블 (레버리지 반영 수익률) — A")
    st.caption("ret_%(levered) = net_ret_% × (size_notional / eq_before)")
    st.dataframe(df_show, use_container_width=True)

    # 에쿼티 커브 & 메트릭
    dates, equity_curve = build_equity_curve(df_log, float(equity))
    metrics = calc_metrics(df_log, equity_curve)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("거래수", metrics["n_trades"])
    c2.metric("Hit-rate", f"{metrics['hit_rate']:.1f}%")
    c3.metric("Avg Win/Loss", f"{metrics['avg_win']:.2f}% / {metrics['avg_loss']:.2f}%")
    c4.metric("Sharpe(연율화)", f"{metrics['sharpe']:.2f}")
    c5.metric("MDD / MAR", f"{metrics['mdd']*100:.2f}% / {metrics['mar']:.2f}")

    if dates and equity_curve and (len(dates) == len(equity_curve)):
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(dates, equity_curve, linewidth=2, label="Equity (A)")
        ax.set_title("Equity Curve (net) — ROLLING (A)")
        ax.grid(True, alpha=0.3); ax.legend()
        st.pyplot(fig)
    else:
        st.warning("에쿼티 커브를 그릴 수 없습니다.")
