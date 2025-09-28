# app_rolling_2020to24.py
# streamlit run app_rolling_2020to24.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --- 네 프로젝트의 모듈들 ---
from connectors import connect_binance
from data_fetch import fetch_futures_4h_klines, fetch_funding_rate
from features import add_features, apply_static_zscore, finalize_preprocessed, GLOBAL_Z_COLS, FEAT_COLS
from similarity import sim_tier3
from blocks import enumerate_blocks
from trading_utils import make_entry_at, make_sl_tp, simulate_trade
from backtest_utils import build_equity_curve, calc_metrics

# ---------------------------
# 고정 기간
# ---------------------------
SCALE_START = pd.Timestamp("2020-01-01 00:00:00")
SCALE_END   = pd.Timestamp("2022-07-01 00:00:00")  # 스케일러 학습 종료
POOL_START  = pd.Timestamp("2025-01-01 00:00:00")
POOL_END    = pd.Timestamp("2025-09-01 00:00:00")
ROLL_START  = pd.Timestamp("2023-01-01 00:00:00")
ROLL_END    = pd.Timestamp("2025-09-01 00:00:00")

# ---------------------------
# 기본 하이퍼파라미터 (ROLLING과 동일)
# ---------------------------
DEFAULTS = dict(
    step_hours=72,
    window_size=18,
    entry_delay_hours=28,
    LO_THR=1.0,
    HI_THR=3.0,
    sim_engine="DTW",      # 후보 검색(sim_tier3) 모드
    w_dtw=0.5,
    sltp_method="ATR",
    k_sl=1.0, k_tp=3.0,
    sl_pct=-0.015, tp_pct=0.03,
    fee_entry=0.0004, fee_exit=0.0005,
    slip_entry=0.0003, slip_exit=0.0005,
    equity=1000.0, risk_pct=0.02, max_leverage=10.0,
    sim_gate_base=0.75, ex_margin_days=5, topN=5,
)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="ROLLING backtest (2020 pool / 2022-24 test)", page_icon="📊", layout="wide")
st.title("📈 BTC ROLLING Backtest — scaler & pool: 2020-01~11, test: 2022-07~2024-01")

colA, colB, colC = st.columns(3)
with colA:
    step_hours = st.number_input("블록 길이(hours)", 24, 168, DEFAULTS["step_hours"], 4)
    window_size = st.number_input("윈도우 길이(bars)", 4, 60, DEFAULTS["window_size"], 1)
    entry_delay_hours = st.number_input("엔트리 지연(hours)", 0, 72, DEFAULTS["entry_delay_hours"], 1)
    sim_engine = st.selectbox("후보 유사도(sim_tier3)", ["DTW","Frechet","Hybrid","Cosine"], index=0)
    w_dtw = st.slider("Hybrid: DTW 가중치", 0.0, 1.0, DEFAULTS["w_dtw"], 0.05)

with colB:
    prefix_metric = st.selectbox("프리픽스 유사도(1D)", ["Cosine","DTW"], index=0)
    LO_THR = st.number_input("LO_THR (%)", 0.0, 20.0, DEFAULTS["LO_THR"], 0.1)
    HI_THR = st.number_input("HI_THR (%)", 0.0, 30.0, DEFAULTS["HI_THR"], 0.1)
    sim_gate_base = st.slider("Similarity gate(base)", 0.0, 1.0, DEFAULTS["sim_gate_base"], 0.01)
    ex_margin_days = st.slider("Exclude margin(days)", 0, 30, DEFAULTS["ex_margin_days"], 1)
    topN = st.slider("후보 topN", 5, 50, DEFAULTS["topN"], 1)

with colC:
    sltp_method = st.radio("SL/TP", ["ATR","FIXED%"], index=0, horizontal=True)
    if sltp_method == "ATR":
        k_sl = st.number_input("k_sl(×ATR)", 0.1, 10.0, DEFAULTS["k_sl"], 0.1)
        k_tp = st.number_input("k_tp(×ATR)", 0.1, 20.0, DEFAULTS["k_tp"], 0.1)
        sl_pct, tp_pct = DEFAULTS["sl_pct"], DEFAULTS["tp_pct"]
    else:
        sl_pct = st.number_input("SL % (음수)", -50.0, 0.0, DEFAULTS["sl_pct"]*100.0, 0.1)/100.0
        tp_pct = st.number_input("TP %", 0.0, 100.0, DEFAULTS["tp_pct"]*100.0, 0.1)/100.0
        k_sl, k_tp = 1.0, 2.0
    fee_entry = st.number_input("Entry fee %", 0.0, 1.0, DEFAULTS["fee_entry"]*100.0, 0.01)/100.0
    fee_exit  = st.number_input("Exit fee %",  0.0, 1.0, DEFAULTS["fee_exit"]*100.0, 0.01)/100.0
    slip_entry = st.number_input("Slippage entry %", 0.0, 1.0, DEFAULTS["slip_entry"]*100.0, 0.01)/100.0
    slip_exit  = st.number_input("Slippage exit %",  0.0, 1.0, DEFAULTS["slip_exit"]*100.0, 0.01)/100.0

st.markdown("---")
colD, colE = st.columns(2)
with colD:
    equity = st.number_input("초기자본(USDT)", 10.0, 1_000_000.0, DEFAULTS["equity"], 10.0)
    risk_pct = st.number_input("포지션 리스크 %", 0.1, 20.0, DEFAULTS["risk_pct"]*100.0, 0.1)/100.0
with colE:
    max_leverage = st.number_input("최대 레버리지(x)", 1.0, 50.0, DEFAULTS["max_leverage"], 1.0)
    fast = st.checkbox("빠른 진행 표시", True)

# ---------------------------
# 유틸 (ROLLING에서 쓰던 형태 유지)
# ---------------------------
def _get_close_at_or_next(df: pd.DataFrame, ts: pd.Timestamp):
    row = df[df["timestamp"] == ts]
    if not row.empty:
        return float(row["close"].iloc[0])
    seg_after = df[df["timestamp"] > ts]
    if not seg_after.empty:
        return float(seg_after["close"].iloc[0])
    return None

def _touch_entry(df: pd.DataFrame, start_ts, end_ts, side: str, target_price: float):
    seg = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    if seg.empty:
        return None, None
    if side == "LONG":
        hit = seg[seg["low"] <= target_price]
        return (hit["timestamp"].iloc[0], float(target_price)) if not hit.empty else (None, None)
    else:
        hit = seg[seg["high"] >= target_price]
        return (hit["timestamp"].iloc[0], float(target_price)) if not hit.empty else (None, None)

def _dtw_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    D = np.full((n+1, m+1), np.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            cost = abs(ai - b[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return float(D[n, m])

def _sim_from_dtw(dist: float) -> float:
    if not np.isfinite(dist):
        return 0.0
    return 1.0 / (1.0 + float(dist))

def _window_is_finite_a(df_window, cols):
    arr = df_window[cols].to_numpy()
    return np.isfinite(arr).all()

def _window_vector_a(df_window, feat_cols, L):
    X = df_window[feat_cols].to_numpy(dtype=float)
    MINMAX_COLS = ['log_ret','atr_z','vol_pct_z']
    for c in MINMAX_COLS:
        if c in feat_cols:
            j = feat_cols.index(c)
            v = X[:, j]
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            X[:, j] = 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.reshape(-1)

def get_candidates_a(df_pool, ref_range, df_ref, feat_cols, step_hours, window_size, sim_mode, w_dtw, topN, ex_margin_days):
    """네 ROLLING 후보 로직과 호환되는 A-variant 후보 탐색."""
    ref_seg = df_ref[(df_ref["timestamp"] >= ref_range[0]) & (df_ref["timestamp"] < ref_range[1])]
    if len(ref_seg) < window_size: return []
    wL = ref_seg.iloc[:window_size]
    if not _window_is_finite_a(wL, feat_cols): return []
    vec_ref = _window_vector_a(wL, feat_cols, L=window_size)

    blocks_pool = enumerate_blocks(df_pool, step_hours=step_hours, window_size=window_size)
    ex_margin = pd.Timedelta(days=ex_margin_days)
    F = len(feat_cols); cand = []
    for b in blocks_pool:
        if not (b["end"] <= ref_range[0] - ex_margin or b["start"] >= ref_range[1] + ex_margin):
            continue
        w = df_pool[(df_pool["timestamp"] >= b["start"]) & (df_pool["timestamp"] < b["end"])]
        if len(w) < window_size: continue
        wL2 = w.iloc[:window_size]
        if not _window_is_finite_a(wL2, feat_cols): continue
        vec_hist = _window_vector_a(wL2, feat_cols, L=window_size)
        if sim_mode == "Cosine":
            if np.allclose(vec_ref, 0) and np.allclose(vec_hist, 0):
                sim = 1.0
            else:
                sim = float(cosine_similarity([vec_ref], [vec_hist])[0][0])
        else:
 
            sim = sim_tier3(vec_ref, vec_hist, L=window_size, F=F, mode=sim_mode, w_dtw=w_dtw)

        cand.append({"start": b["start"], "end": b["end"], "sim": sim})
    cand.sort(key=lambda x: x["sim"], reverse=True)
    return cand[:topN]

def _risk_size_with_leverage(entry_price, sl, eq_run, risk_pct, max_leverage):
    if eq_run is None or eq_run <= 0: return 0.0, 0.0, False
    size_notional = float(eq_run) * float(max_leverage)
    used_lev = float(max_leverage)
    cap_hit = True
    return size_notional, used_lev, cap_hit

# ---------------------------
# 데이터 로드 & 전처리 (여기서 client를 제대로 씀)
# ---------------------------
st.caption("데이터 로드 중…")
client = connect_binance()  # ← 반드시 Client 인스턴스여야 함

df_raw = fetch_futures_4h_klines(client, start_time=str(SCALE_START.date()))
df_funding = fetch_funding_rate(client, start_time=str(SCALE_START.date()))

st.caption("피처 생성 및 스케일러(2020-01~11) 적용…")
df_feat = add_features(df_raw, df_funding)
df_scaled = apply_static_zscore(df_feat.copy(), GLOBAL_Z_COLS, SCALE_END)
df_scaled = finalize_preprocessed(df_scaled, window_size)

pool_df = df_scaled[(df_scaled["timestamp"] >= POOL_START) & (df_scaled["timestamp"] < POOL_END)].reset_index(drop=True)
if len(pool_df) < window_size:
    st.error("POOL 구간(2020-01~11) 데이터가 부족합니다.")
    st.stop()

roll_base = df_scaled[(df_scaled["timestamp"] >= (ROLL_START - pd.Timedelta(hours=step_hours))) &
                      (df_scaled["timestamp"] <= ROLL_END)].reset_index(drop=True)
if len(roll_base) < window_size:
    st.error("백테스트 구간(2022-07~2024-01) 데이터가 부족합니다.")
    st.stop()

blocks = enumerate_blocks(roll_base, step_hours=step_hours, window_size=window_size)
start_idx = None; end_idx = None
for i in range(1, len(blocks)):
    if blocks[i]["start"] >= ROLL_START and start_idx is None:
        start_idx = i
    if blocks[i]["start"] >= ROLL_END and end_idx is None:
        end_idx = i; break
if start_idx is None: st.error("시작 블록 없음"); st.stop()
if end_idx is None: end_idx = len(blocks)

# ---------------------------
# 백테스트 루프 (ROLLING 로직 그대로)
# ---------------------------
trade_logs = []
stepTD = pd.Timedelta(hours=step_hours)
delayTD = pd.Timedelta(hours=entry_delay_hours)
pbar = st.progress(0)
total_iter = max(1, (end_idx - start_idx))
eq_run = float(equity)

for j, bp in enumerate(range(start_idx, end_idx)):
    ref_b = blocks[bp - 1]; pred_b = blocks[bp]
    t_entry = pred_b["start"] + delayTD
    if t_entry > pred_b["end"]:
        pbar.progress(int(100*(j+1)/total_iter)); continue

    # 후보: 2020 풀에서만 검색 (sim_tier3)
    cands = get_candidates_a(
        df_pool=pool_df,
        ref_range=(ref_b["start"], ref_b["end"]),
        df_ref=df_scaled,
        feat_cols=FEAT_COLS,
        step_hours=step_hours,
        window_size=window_size,
        sim_mode=sim_engine,
        w_dtw=w_dtw,
        topN=topN,
        ex_margin_days=ex_margin_days,
    )
    if not cands:
        pbar.progress(int(100*(j+1)/total_iter)); continue

    # 각 후보의 다음 72h 변화율
    results = []
    for f in cands:
        next_start = f["end"]; next_end = next_start + stepTD
        df_next = pool_df[(pool_df["timestamp"] >= next_start) & (pool_df["timestamp"] < next_end)]
        if len(df_next) < window_size: continue
        closes = df_next["close"].to_numpy(); baseC = float(closes[0])
        pct_c = (closes - baseC) / baseC * 100.0
        results.append({
            "sim": f["sim"], "next_start": next_start, "next_end": next_end,
            "pct": pct_c, "df_next": df_next.reset_index(drop=True), "base_close": baseC
        })
    if not results:
        pbar.progress(int(100*(j+1)/total_iter)); continue

    # 현재 프리픽스 (pred_b.start ~ t_entry)
    pred_seg = roll_base[(roll_base["timestamp"] >= pred_b["start"]) & (roll_base["timestamp"] <= t_entry)]
    if len(pred_seg) == 0:
        pbar.progress(int(100*(j+1)/total_iter)); continue
    base_cur = float(pred_seg["close"].iloc[0])
    a = ((pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)
    L = len(a)

    # 프리픽스 매칭 (1D Cosine or 1D DTW)
    best = None
    for r in results:
        b = np.array(r["pct"], dtype=float)[:L]
        if prefix_metric == "Cosine":
            sim_shape = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) else float(cosine_similarity([a],[b])[0][0])
        else:
            dist = _dtw_distance_1d(a, b)
            sim_shape = _sim_from_dtw(dist)
        if (best is None) or (sim_shape > best["sim"]):
            best = {"sim": sim_shape, "flow": r}
    if best is None:
        pbar.progress(int(100*(j+1)/total_iter)); continue

    # 후행부 특성
    hist_full = np.array(best["flow"]["pct"], dtype=float)
    base_now = float(hist_full[L - 1])
    fut = hist_full[L - 1:] - base_now
    idx_max = int(np.argmax(fut)) if fut.size > 0 else 0
    idx_min = int(np.argmin(fut)) if fut.size > 0 else 0
    max_up = float(np.max(fut)) if fut.size > 0 else 0.0
    min_dn = float(np.min(fut)) if fut.size > 0 else 0.0

    # 레짐/게이트/임계
    ext_start = pred_b["start"] - pd.Timedelta(hours=48)
    prefix_end = min(t_entry, pred_b["end"])
    ext_seg = roll_base[(roll_base["timestamp"] >= ext_start) & (roll_base["timestamp"] <= prefix_end)].reset_index(drop=True)
    used_ext = (len(ext_seg) >= 2)
    seg = ext_seg if used_ext else pred_seg
    anchor = float(seg["close"].iloc[0]); last = float(seg["close"].iloc[-1])
    ret_pct = (last / anchor - 1.0) * 100.0
    cutoff = (-1.0 if used_ext else 0.0)
    regime_down = (ret_pct < cutoff)

    sim_gate = float(sim_gate_base) + (0.05 if regime_down else 0.0)
    LO_THR_USE = LO_THR + (0.5 if regime_down else 0.0)
    HI_THR_USE = HI_THR + (0.5 if regime_down else 0.0)

    # 방향 결정 (ROLLING 규칙)
    side = "HOLD"
    if best["sim"] >= sim_gate:
        mag_up = abs(max_up); mag_dn = abs(min_dn); mag = max(mag_up, mag_dn)
        if mag >= LO_THR_USE:
            if regime_down and (mag_up >= mag_dn):
                side = "HOLD"
            else:
                side = "LONG" if mag_up >= mag_dn else "SHORT"

    # 진입 계산
    entry_time = entry_price = entry_target = None
    SL = TP = None
    if side in ("LONG","SHORT"):
        if max(abs(max_up), abs(min_dn)) >= HI_THR_USE:
            # HI 이상: t_entry에서 다음봉 시가
            et0, ep0 = make_entry_at(roll_base, t_entry, rule="다음봉 시가")
            if et0 is not None and et0 < t_entry:
                seg_after = roll_base[roll_base["timestamp"] > t_entry]
                if not seg_after.empty:
                    et0, ep0 = seg_after["timestamp"].iloc[0], float(seg_after["open"].iloc[0])
            entry_time, entry_price = et0, ep0
        else:
            # LO~HI: 되돌림 리밋 매핑 → 터치 체결
            df_next_best = best["flow"]["df_next"]
            base_hist_close = float(best["flow"]["base_close"])
            cur_28h_close = _get_close_at_or_next(roll_base, t_entry)
            if cur_28h_close is not None and len(df_next_best) > 0:
                if side == "LONG":
                    end_k = min((L - 1) + idx_max, len(df_next_best) - 1)
                    lows_slice = df_next_best["low"].iloc[:end_k + 1].to_numpy(dtype=float)
                    if lows_slice.size > 0:
                        low_min = float(np.min(lows_slice))
                        drop_pct = (low_min / base_hist_close - 1.0) * 100.0  # 음수
                        mag_adj = max(0.0, abs(drop_pct))
                        entry_target = cur_28h_close * (1.0 - mag_adj/100.0)
                        entry_time, entry_price = _touch_entry(roll_base, t_entry, pred_b["end"], "LONG", entry_target)
                else:
                    end_k = min((L - 1) + idx_min, len(df_next_best) - 1)
                    highs_slice = df_next_best["high"].iloc[:end_k + 1].to_numpy(dtype=float)
                    if highs_slice.size > 0:
                        high_max = float(np.max(highs_slice))
                        up_pct = (high_max / base_hist_close - 1.0) * 100.0
                        mag_adj = max(0.0, abs(up_pct))
                        entry_target = cur_28h_close * (1.0 + mag_adj/100.0)
                        entry_time, entry_price = _touch_entry(roll_base, t_entry, pred_b["end"], "SHORT", entry_target)

    # SL/TP
    atr_ref = None
    if entry_time is not None:
        row_at = roll_base[roll_base["timestamp"] == entry_time]
        if not row_at.empty and row_at["atr"].notna().any():
            atr_ref = float(row_at["atr"].fillna(method='ffill').iloc[0])
    if side in ("LONG","SHORT") and (entry_time is not None) and (entry_price is not None):
        SL, TP = make_sl_tp(
            entry_price, side,
            method=("ATR" if sltp_method=="ATR" else "FIXED"),
            atr=atr_ref, sl_pct=sl_pct, tp_pct=tp_pct, k_sl=k_sl, k_tp=k_tp, tick_size=0.0
        )
    else:
        side = "HOLD"

    # 사이징 & 시뮬
    size = used_lev = 0.0; cap_hit = False
    exit_time = exit_price = gross_ret = net_ret = None
    if side in ("LONG","SHORT") and (entry_time is not None) and (entry_price is not None) and (SL is not None):
        size, used_lev, cap_hit = _risk_size_with_leverage(entry_price, SL, eq_run, risk_pct, max_leverage)
        exit_time, exit_price, gross_ret, net_ret = simulate_trade(
            df=roll_base, start_ts=pred_b["start"], end_ts=pred_b["end"],
            side=side, entry_time=entry_time, entry_price=float(entry_price),
            sl=SL, tp=TP,
            fee_entry=fee_entry, fee_exit=fee_exit, slip_entry=slip_entry, slip_exit=slip_exit,
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

    trade_logs.append({
        "pred_start": pred_b["start"], "pred_end": pred_b["end"], "t_entry": t_entry,
        "side": side, "entry_time": entry_time, "entry": entry_price, "entry_target": entry_target,
        "SL": SL, "TP": TP, "exit_time": exit_time, "exit": exit_price,
        "gross_ret_%": gross_ret, "net_ret_%": net_ret,
        "size_notional": size, "used_lev": used_lev, "cap_hit": cap_hit,
        "eq_before": eq_before, "eq_after": eq_run, "pnl_usd": pnl_usd, "ret_%(levered)": ret_equity_pct,
        "sim_shape": best["sim"],
    })

    if fast:
        pbar.progress(int(100*(j+1)/total_iter))

# ---------------------------
# 결과 출력
# ---------------------------
if not trade_logs:
    st.info("거래 로그가 없습니다. (조건 미충족/HOLD 등)")
    st.stop()

df_log = pd.DataFrame(trade_logs).sort_values("pred_start").reset_index(drop=True)

st.markdown("### 결과 테이블 (레버리지 반영 수익률)")
cols = ["pred_start","pred_end","t_entry","side","entry_time","entry","entry_target","SL","TP",
        "size_notional","used_lev","cap_hit","pnl_usd","ret_%(levered)","eq_before","eq_after",
        "exit_time","exit","sim_shape"]
st.dataframe(df_log[[c for c in cols if c in df_log.columns]], use_container_width=True)

# 메트릭 & 에쿼티
try:
    dates, equity_curve = build_equity_curve(df_log, float(equity))
    metrics = calc_metrics(df_log, equity_curve)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("거래수", metrics.get("n_trades", 0))
    c2.metric("Hit-rate", f"{metrics.get('hit_rate', 0.0):.1f}%")
    c3.metric("Avg Win/Loss", f"{metrics.get('avg_win', 0.0):.2f}% / {metrics.get('avg_loss', 0.0):.2f}%")
    c4.metric("Sharpe(연율화)", f"{metrics.get('sharpe', 0.0):.2f}")
    c5.metric("MDD / MAR", f"{metrics.get('mdd', 0.0)*100:.2f}% / {metrics.get('mar', 0.0):.2f}")

    if dates and equity_curve and (len(dates) == len(equity_curve)):
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(dates, equity_curve, linewidth=2)
        ax.set_title("Equity Curve (net)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.warning("에쿼티 커브를 그릴 수 없습니다.")
except Exception as e:
    st.warning(f"메트릭 계산 중 오류: {e}")
