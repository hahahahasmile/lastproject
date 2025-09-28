# === Prelude: 한글 폰트/음수 디폴트 적용 (UI 없음) ===
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False      # 음수 기호 깨짐 방지
# === End Prelude ===


# === ui/app.py (ATR-only, sl_pct/tp_pct 제거 버전) ===
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
st.set_page_config(page_title="BTC 패턴매칭 전략 스튜디오", page_icon="📊", layout="wide")
st.title("📈 BTC 패턴매칭 전략 스튜디오")

# ---------------------------
# 공통 하이퍼파라미터
# ---------------------------
step_hours = 72
window_size = 18
ENTRY_DELAY_HOURS = 28
ENTRY_RULE_FIXED = "다음봉 시가"

LO_THR = 1.0
HI_THR = 3.0
sim_gate_base = 0.75

# ---------------------------
# 상단 UI
# ---------------------------

colA, colB, colC = st.columns(3)

with colA:
    sim_mode = st.radio(
        "모드",
        ["NOW-상승", "BT-상승", "NOW-하락/횡보", "BT-하락/횡보", "LIVE"],
        index=0, horizontal=True,
        help="NOW-상승: 단일·28h / BT-상승: 연속BT / NOW-하락·횡보: 단일·28h / BT-하락·횡보: 연속BT / LIVE: 주문 미리보기"
    )

    # 공통 디폴트 (필요시 각 모드에서 재설정)
    sim_engine = "DTW"   # ROLLING 계열에서만 사용
    w_dtw = 0.5          # Hybrid 제거되었지만 호출 시 인자형 유지(무시됨)

    # SL/TP은 기본 ATR 사용. (ROLLING/ROLLING(B)에서 FIXED 제거)
    sltp_method = "ATR"
    k_sl = 1.0
    k_tp = 2.5
    # (sl_pct / tp_pct 완전 제거)

    fee_entry = 0.0004
    fee_exit  = 0.0005
    slip_entry = 0.0003
    slip_exit  = 0.0005

    equity = 1000.0
    fast = True
    max_leverage = 10.0

# ---- ROLLING 상단 설정 패널(수정: FIXED 제거, fast 강제 True) ----
if sim_mode == "BT-상승":
    with colA:
        sim_engine = st.selectbox(
            "유사도 방식",
            ["DTW", "Cosine"],
            index=0,
            help="과거 구간과의 유사도 계산 메트릭. DTW(동적 타임워핑) 또는 Cosine(코사인 유사도)만 허용."
        )
        st.caption(f"|Δ|≥{HI_THR:.1f}% → 즉시 / {LO_THR:.1f}%~<{HI_THR:.1f}% → 되돌림 / <{LO_THR:.1f}% → HOLD")

    with colB:
        k_sl = st.number_input("SL × ATR", 0.1, 10.0, 1.0, 0.1, help="손절폭 = k_sl×ATR. 예) 1.0 → 엔트리 ± 1×ATR.")
        k_tp = st.number_input("TP × ATR", 0.1, 20.0, 2.5, 0.1, help="익절폭 = k_tp×ATR. 예) 3.0 → 엔트리 ± 3×ATR.")

        fee_entry = st.number_input("진입 수수료(%)", 0.0, 1.0, 0.04, 0.01, help="백테스트 체결을 현실화하기 위한 가정 수수료. 0.04는 0.04%.") / 100.0
        fee_exit  = st.number_input("청산 수수료(%)", 0.0, 1.0, 0.05, 0.01, help="백테스트 체결 현실화용 가정 수수료.") / 100.0
        slip_entry = st.number_input("진입 슬리피지(%)", 0.0, 0.5, 0.03, 0.01, help="체결 미끄러짐 가정치(%).") / 100.0
        slip_exit  = st.number_input("청산 슬리피지(%)", 0.0, 0.5, 0.05, 0.01, help="체결 미끄러짐 가정치(%).") / 100.0

    with colC:
        equity = st.number_input("가상 Equity (USDT)", 10.0, value=1000.0, step=10.0, help="백테스트/포지션 사이징에 사용하는 가상의 계정 잔고(USDT).")
        max_leverage = st.number_input("최대 레버리지(x)", 1.0, 50.0, 10.0, 1.0, help="사이징 계산 시 사용할 레버리지 상한(실체결 한도 아님).")

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

# ---------------------------
# NOW
# ---------------------------
if sim_mode == "NOW-상승":
    st.subheader("NOW-상승: 28h 지연 엔트리 · 1회 거래 (태그별 전략 명시 포함)")

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
        ENTRY_DELAY_BARS = max(1, int(np.ceil(ENTRY_DELAY_HOURS / 4)))  # 28h -> 7 bars
        base_close_28h = float(df_next["close"].iloc[ENTRY_DELAY_BARS - 1]) \
            if len(df_next) >= ENTRY_DELAY_BARS else float(df_next["close"].iloc[-1])
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
    ENTRY_DELAY_BARS = int(np.ceil(ENTRY_DELAY_HOURS / 4.0))  # 28h -> 7
    L_use = ENTRY_DELAY_BARS + 1  # 0~7 포함 → 8개 고정
    a = a[:min(L_use, len(a))]
    L = len(a)
    # 프리픽스 최고 후보 선정
    best = None
    for r in results:
        b = np.array(r["pct"], dtype=float)[:L]
        sim_shape = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) else float(cosine_similarity([a], [b])[0][0])
        if (best is None) or (sim_shape > best["sim"]):
            best = {"sim": sim_shape, "flow": r}

    df_best_next = best["flow"]["df_next"]
    base_hist_close = best["flow"].get("base_close_28h",
                                       best["flow"].get("base_close",
                                                        (float(df_best_next["close"].iloc[L-1]) if len(df_best_next) >= L and L>0 else float(df_best_next["close"].iloc[-1]))))

    # 표
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
    ax.set_title("NOW-상승: 28h 기준 · 진행 vs 매칭 (원시%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.caption("세로 점선: 28h(엔트리 기준), 가로 점선 LO/HI: 중·강 임계값, 곡선: 프리픽스/후보 원시%")
    st.pyplot(fig)

    # ---------------- NOW: 시나리오 비교 ----------------
    fut = hist_full[min(int(np.ceil(ENTRY_DELAY_HOURS/4.0)), len(hist_full) - 1):] - hist_full[min(int(np.ceil(ENTRY_DELAY_HOURS/4.0)), len(hist_full) - 1)]
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

    STRAT_DESC = {
        "A": "강한 상승: HI_THR_USE 이상 상승 우위 → 다음봉 시가 진입",
        "B": "강한 하락: HI_THR_USE 이상 하락 우위 → 다음봉 시가 진입",
        "C": "중간 상승: LO~HI, 상승 우위 → 되돌림 리밋가 (가정값)",
        "C′": "중간 하락: LO~HI, 하락 우위 → 되돌림 리밋가 (가정값)",
        "D": "항상 HOLD",
        "E": "약함/미달 → HOLD"
    }

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

    force_mode = st.checkbox(
        "강제 가정으로 계산(조건 불충족이어도 값 채움)",
        value=True,
        help="C/C′ 되돌림 리밋이 실제로 터치되지 않아도, 가정값으로 진입가를 채워 SL/TP를 계산해 비교합니다."
    )

    cur_28h_close = _get_close_at_or_next(df_full, pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS))
    base_hist_close_local = base_hist_close

    def scenario_row_now(tag: str, plan_side: str, cond_ok: bool, force_mode_local: bool):
        note = ""
        t_entry_local = pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS)
        if tag in ("D", "E"):
            return {"scenario": tag, "설명": STRAT_DESC.get(tag, ""), "side": "HOLD", "t_entry": t_entry_local,
                    "entry_time": None, "entry_price": None, "SL": None, "TP": None,
                    "cond_ok": cond_ok, "touched": False, "forced": False, "used_rule": "HOLD", "note": "항상 HOLD"}

        if tag in ("A", "B"):
            et0, ep0 = make_entry_at(df_full, t_entry_local, rule=ENTRY_RULE_FIXED)
            if et0 is not None and et0 < t_entry_local:
                et0, ep0 = next_open_after_local(t_entry_local)
            entry_time, entry_price = et0, ep0
            touched = False; forced = False; used_rule = "다음봉 시가"
        else:
            entry_time = None; entry_price = None; touched = False; forced = False; used_rule = "리밋가(가정)"
            if cur_28h_close is not None and len(df_best_next) > 0:
                if tag == "C":
                    target = compute_limit_target_local("LONG", df_best_next, L, idx_max, idx_min, cur_28h_close, base_hist_close_local)
                    if target is not None:
                        et, ep = _touch_entry(df_full, t_entry_local, pred_end, "LONG", target)
                        if et is not None:
                            entry_time, entry_price = et, ep; touched = True
                        elif force_mode_local:
                            entry_time, entry_price = t_entry_local, float(target); forced = True
                else:
                    target = compute_limit_target_local("SHORT", df_best_next, L, idx_max, idx_min, cur_28h_close, base_hist_close_local)
                    if target is not None:
                        et, ep = _touch_entry(df_full, t_entry_local, pred_end, "SHORT", target)
                        if et is not None:
                            entry_time, entry_price = et, ep; touched = True
                        elif force_mode_local:
                            entry_time, entry_price = t_entry_local, float(target); forced = True
            if (entry_price is None) and force_mode_local:
                et0, ep0 = make_entry_at(df_full, t_entry_local, rule=ENTRY_RULE_FIXED)
                if et0 is not None and et0 < t_entry_local:
                    et0, ep0 = next_open_after_local(t_entry_local)
                entry_time, entry_price = et0, ep0
                note += "리밋 불가→시가 대체; "

        SL = TP = None
        side_out = plan_side if tag in ("C", "C′") else ("LONG" if tag == "A" else "SHORT")
        if entry_time is not None and entry_price is not None:
            row_at = df_full[df_full["timestamp"] == entry_time]
            atr_ref_local = float(row_at["atr"].fillna(method='ffill').iloc[0]) if (not row_at.empty and row_at["atr"].notna().any()) else None
            # ATR 고정 (sl_pct/tp_pct 사용 안함)
            SL, TP = make_sl_tp(entry_price, side_out, method="ATR", atr=atr_ref_local,
                                 sl_pct=None, tp_pct=None, k_sl=k_sl, k_tp=k_tp, tick_size=0.0)
        else:
            side_out = "HOLD"

        return {"scenario": tag, "설명": STRAT_DESC.get(tag, ""), "side": side_out, "t_entry": t_entry_local,
                "entry_time": entry_time, "entry_price": entry_price, "SL": SL, "TP": TP,
                "cond_ok": cond_ok, "touched": touched, "forced": forced, "used_rule": used_rule, "note": note}

    rows = []
    cond_A = (mag >= HI_THR_USE) and up_win and (not regime_down)
    rows.append(scenario_row_now("A", "LONG", cond_A, force_mode))
    cond_B = (mag >= HI_THR_USE) and dn_win
    rows.append(scenario_row_now("B", "SHORT", cond_B, force_mode))
    cond_C = (LO_THR_USE <= mag < HI_THR_USE) and up_win and (not regime_down)
    rows.append(scenario_row_now("C", "LONG", cond_C, force_mode))
    cond_Cp = (LO_THR_USE <= mag < HI_THR_USE) and dn_win
    rows.append(scenario_row_now("C′", "SHORT", cond_Cp, force_mode))
    cond_D = (LO_THR_USE <= mag < HI_THR_USE) and up_win and regime_down
    rows.append(scenario_row_now("D", "HOLD", cond_D, force_mode))
    cond_E = (mag < LO_THR_USE) or (best["sim"] < sim_gate)
    rows.append(scenario_row_now("E", "HOLD", cond_E, force_mode))

    if st.button(
        "시나리오 계산",
        help="프리픽스(0~28h)와 매칭 후보를 바탕으로 A~E 시나리오별 진입/SL/TP·거리(Δ)·퍼센트를 표로 계산합니다."
    ):
        df_scn = pd.DataFrame(rows)

        def _delta(row):
            ep = row.get("entry_price"); sl = row.get("SL"); tp = row.get("TP")
            if ep is None or sl is None or tp is None:
                return pd.Series([None, None, None, None])
            sl_d = abs(ep - sl); tp_d = abs(tp - ep)
            sl_pct_v = (sl_d / ep) * 100.0; tp_pct_v = (tp_d / ep) * 100.0
            return pd.Series([sl_d, tp_d, sl_pct_v, tp_pct_v])

        df_scn[["SL_Δ","TP_Δ","SL_%","TP_%"]] = df_scn.apply(_delta, axis=1)
        show_cols = ["scenario","설명","side","t_entry","entry_time","entry_price","used_rule","cond_ok","touched","forced","SL","TP","SL_Δ","TP_Δ","SL_%","TP_%","note"]
        df_scn = df_scn[[c for c in show_cols if c in df_scn.columns]]
        st.dataframe(df_scn, use_container_width=True)

# ---------------------------
# LIVE (실거래)
# ---------------------------
elif sim_mode == "LIVE":
    st.subheader("LIVE: 실거래 (메인넷)")
    df_full = df_full_static

    entry_rule = ENTRY_RULE_FIXED
    sltp_method = "ATR"; k_sl, k_tp = 1.0, 2.5

    # ── 1) 계정/지갑 UI ─────────────────────────────────────────────
    with st.expander("💳 계정 · 선물 지갑 (메인넷)", expanded=True):
        tclient = connect_binance_trade()
        trade_symbol = st.text_input("거래 심볼", value="BTCUSDT", help="선물 심볼. 예: BTCUSDT (USDT 무기한).")
        leverage = st.number_input("레버리지(x)", 1, 100, 10, 1, help="거래소에 설정되는 레버리지 값.")
        margin_mode = st.radio("마진 모드", ["교차(Cross)", "격리(Isolated)"], index=0, horizontal=True, help="교차: 계정 전체 증거금 공유 / 격리: 포지션별 증거금 분리.")
        use_cross = (margin_mode == "교차(Cross)")
        size_pct = st.slider("사이즈 % (가용잔고 기준)", 0.1, 100.0, 2.0, 0.1, help="가용 잔고×레버리지에 대한 진입 노션럴 비율.")

        # 신호 소스: NOW-상승 / NOW-하락·횡보 / 수동(HOLD)
        signal = st.radio("신호 소스", ["NOW-상승(Long)", "NOW-하락/횡보(Short)", "수동(HOLD)"], index=0, horizontal=True)

        # 슬리피지(미리보기용)
        slip_entry = st.number_input("진입 슬리피지(%)", 0.0, 1.0, 0.03, 0.01) / 100.0

        bals = get_futures_balances(tclient)
        colb1, colb2 = st.columns(2)
        colb1.metric("USDT Wallet", f"{bals['wallet_balance']:.2f}")
        colb2.metric("USDT Available", f"{bals['available_balance']:.2f}")

    # ── 2) 엔트리 시점 산출 ─────────────────────────────────────────
    (ref_start, ref_end), (pred_start, pred_end) = pick_blocks(df_full['timestamp'].iloc[-1], step_hours=step_hours)
    t_entry = pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS)

    entry_time, entry_price = make_entry_at(df_full, t_entry, rule=entry_rule)
    if (entry_time is not None) and (entry_time < t_entry):
        seg_after = df_full[df_full["timestamp"] > t_entry]
        if not seg_after.empty:
            entry_time = seg_after["timestamp"].iloc[0]
            entry_price = float(seg_after["open"].iloc[0])

    # 데이터 점검
    if entry_time is None or entry_price is None:
        st.error("엔트리 기준 봉을 찾지 못했습니다. 데이터가 부족하거나 시점이 맞지 않습니다.")
        st.stop()

    atr_ref = float(df_full.loc[df_full["timestamp"] == entry_time, "atr"].fillna(method='ffill').iloc[0]) \
        if entry_time is not None else None
    if atr_ref is None or np.isnan(atr_ref):
        st.error("ATR 데이터가 없습니다. SL/TP 계산 불가.")
        st.stop()

    # ── 3) 거래소 세팅/필터 ────────────────────────────────────────
    tclient2 = connect_binance_trade()
    ensure_leverage_and_margin(tclient2, symbol=trade_symbol, leverage=int(leverage), cross=use_cross)
    tick_size, qty_step = get_symbol_filters(tclient2, symbol=trade_symbol)

    def _round_price(p):
        if p is None: return None
        if tick_size and tick_size > 0:
            return np.floor(p / tick_size) * tick_size
        return p

    def _round_qty(q):
        if q is None: return None
        if qty_step and qty_step > 0:
            return np.floor(q / qty_step) * qty_step
        return q

    # ── 4) 신호 → 포지션 방향 결정 ─────────────────────────────────
    if signal.startswith("NOW-상승"):
        side = "LONG"
    elif signal.startswith("NOW-하락/횡보"):
        side = "SHORT"
    else:
        side = "HOLD"

    # ── 5) 슬리피지 반영 체결가 & SL/TP 계산(방향 보장) ─────────────
    if side == "LONG":
        fill_price = entry_price * (1.0 + slip_entry)
    elif side == "SHORT":
        fill_price = entry_price * (1.0 - slip_entry)
    else:
        fill_price = entry_price  # HOLD일 때도 참조용

    if side in ("LONG", "SHORT"):
        sl_raw, tp_raw = make_sl_tp(
            fill_price, side, method=sltp_method, atr=atr_ref,
            sl_pct=None, tp_pct=None, k_sl=k_sl, k_tp=k_tp, tick_size=0.0
        )
        # 방향 보장
        if side == "LONG":
            if sl_raw is not None: sl_raw = min(sl_raw, fill_price)
            if tp_raw is not None: tp_raw = max(tp_raw, fill_price)
        else:  # SHORT
            if sl_raw is not None: sl_raw = max(sl_raw, fill_price)
            if tp_raw is not None: tp_raw = min(tp_raw, fill_price)
    else:
        sl_raw = tp_raw = None

    # 거래소 규격 라운딩
    entry_rounded = _round_price(fill_price)
    sl_rounded = _round_price(sl_raw) if sl_raw is not None else None
    tp_rounded = _round_price(tp_raw) if tp_raw is not None else None

    # ── 6) 수량 산출(가용잔고 기반) + 라운딩 ────────────────────────
    avail = bals.get("available_balance", 0.0) or 0.0
    notional = max(0.0, avail) * float(leverage) * (float(size_pct) / 100.0)
    qty_live = (notional / entry_rounded) if (entry_rounded and entry_rounded > 0) else 0.0
    qty_live = _round_qty(qty_live)

    # ── 7) 사전 검증(표시만, 자동조정 없음) ─────────────────────────
    issues = []
    if side == "LONG":
        if (sl_rounded is None) or (tp_rounded is None) or not (sl_rounded < entry_rounded < tp_rounded):
            issues.append("LONG 조건 위반: SL < Entry < TP 가 보장되지 않았습니다.")
    elif side == "SHORT":
        if (sl_rounded is None) or (tp_rounded is None) or not (tp_rounded < entry_rounded < sl_rounded):
            issues.append("SHORT 조건 위반: TP < Entry < SL 가 보장되지 않았습니다.")
    if qty_live is None or qty_live <= 0:
        issues.append("수량이 0입니다. 잔고/레버리지/사이즈%를 확인하세요.")
    if tick_size is None or qty_step is None:
        issues.append("거래소 필터 조회 실패(tick_size/qty_step). 주문이 거절될 수 있습니다.")

    # ── 8) 미리보기 ─────────────────────────────────────────────────
    st.markdown("### 📌 주문 미리보기")
    colp1, colp2, colp3, colp4, colp5 = st.columns(5)
    colp1.metric("Side", side)
    colp2.metric("Entry(라운딩)", f"{(entry_rounded or 0):.2f}")
    colp3.metric("SL(라운딩)", f"{(sl_rounded or 0):.2f}")
    colp4.metric("TP(라운딩)", f"{(tp_rounded or 0):.2f}")
    colp5.metric("수량(계약, 라운딩)", f"{(qty_live or 0):.6f}")
    st.caption("슬리피지 적용 → 가격/수량 라운딩 → 검증 순으로 계산됩니다. 파라미터(ATR·임계치)는 기존과 동일.")

    if issues:
        for msg in issues:
            st.warning(f"⚠ {msg}")
    else:
        st.success("검증 통과: 형식/방향 OK (주문 가능 상태)")

    # ── 9) 슬리피지 캡 + 동시 발주 버튼 ─────────────────────────────
    st.divider()
    colo1, colo2 = st.columns(2)
    with colo1:
        slip_cap_pct = st.number_input("슬리피지 캡(%)", 0.0, 5.0, 0.5, 0.1, help="시장가 체결 보호. 예상 진입가 대비 허용 편차 한도.")
    with colo2:
        ttl_min = st.number_input("신호 유효시간(분)", 1, 180, 30, 1, help="발주 버튼 활성화 유효시간(현재 시각 - 엔트리 기준 시각)")

    # 신호 TTL 확인 (간단히 현재 df의 마지막 시각을 '현재'로 간주)
    now_like = df_full["timestamp"].iloc[-1]
    ttl_ok = (now_like - t_entry) <= pd.Timedelta(minutes=float(ttl_min))

    # 슬리피지 캡 사전 검증: 최신 종가 기준으로 단순 비교(실시간 호가 대신 대용)
    last_px = float(df_full["close"].iloc[-1])
    exp_entry = entry_rounded
    slip_obs_pct = abs(exp_entry - last_px) / exp_entry * 100.0 if exp_entry else 0.0
    if slip_obs_pct > slip_cap_pct:
        issues.append(f"슬리피지 캡 초과: 관측 {slip_obs_pct:.2f}% > 허용 {slip_cap_pct:.2f}%")

    # 버튼 상태
    can_order = (side in ("LONG","SHORT")) and (not issues) and ttl_ok

    st.markdown("### 🧾 사전 체크리스트")
    st.write({
        "TTL OK": ttl_ok,
        "Side": side,
        "tick_size/qty_step OK": (tick_size is not None and qty_step is not None),
        "Qty>0": (qty_live is not None and qty_live > 0),
        "SL/TP 방향 보장": ( (side=="LONG" and sl_rounded is not None and tp_rounded is not None and sl_rounded<entry_rounded<tp_rounded)
                           or (side=="SHORT" and sl_rounded is not None and tp_rounded is not None and tp_rounded<entry_rounded<sl_rounded) )
    })

    order_btn = st.button("🟢 동시 발주 (Entry+SL+TP)", disabled=not can_order, help="검증 통과 시에만 활성화. GTC로 한 번에 제출.")

    if order_btn:
        try:
            resp = place_futures_market_bracket(
                tclient2,
                symbol=trade_symbol,
                side=side,
                qty=float(qty_live),
                entry=float(entry_rounded),
                sl=float(sl_rounded),
                tp=float(tp_rounded),
                time_in_force="GTC"
            )
            st.success("발주 성공: 브래킷 세트가 제출되었습니다.")
            st.json(resp)
        except Exception as e:
            st.error(f"발주 실패: {e}")

    # ── 10) 포지션/미체결 확인 패널 ────────────────────────────────
    st.markdown("### 📌 현재 포지션 / 미체결")
    try:
        pos = get_futures_positions(tclient2)
        if not pos:
            st.info("열린 포지션이 없습니다.")
        else:
            cols_show = ["symbol","positionAmt","entryPrice","unRealizedProfit","leverage","liquidationPrice","markPrice"]
            df_pos = pd.DataFrame(pos)
            df_pos = df_pos[[c for c in cols_show if c in df_pos.columns]]
            st.dataframe(df_pos, use_container_width=True)
    except Exception as e:
        st.warning(f"포지션 조회 실패: {e}")


# ---------------------------
# ROLLING (원본) — FIXED 제거 & fast 강제 True
# ---------------------------
elif sim_mode == "BT-상승":
    st.subheader("BT-상승: 28h 지연 엔트리 · 블록당 1회 거래 백테스트 (Static only, ATR 고정, fast 모드)")

    # 공통 파라미터
    topN = 5  # fast 강제
    exd = 10  # fast 강제
    stepTD = pd.Timedelta(hours=step_hours)
    delayTD = pd.Timedelta(hours=ENTRY_DELAY_HOURS)

    # 백테스트 예측 구간 시작(현재 분석 구간)
    ROLL_START = pd.Timestamp("2025-01-01 00:00:00")

    # 블록 시퀀스 기준(공통) — static으로 시간축 고정
    df_roll_base = df_full_static[df_full_static["timestamp"] >= (ROLL_START - pd.Timedelta(hours=72))].reset_index(drop=True)
    if len(df_roll_base) < window_size:
        st.warning("BT-상승: 데이터 부족")
        st.stop()

    blocks_all = enumerate_blocks(df_roll_base, step_hours=step_hours, window_size=window_size)
    start_idx = None
    for i in range(1, len(blocks_all)):
        if blocks_all[i]["start"] >= ROLL_START:
            start_idx = i
            break
    if start_idx is None:
        st.warning("BT-상승: 2025년 이후 pred 블록 없음")
        st.stop()

    # 후보 기간 정책 (static만 사용): 2025-01-01 이후
    hist_start_static = pd.Timestamp("2025-01-01 00:00:00")

    # 단일 variant 평가 함수 (ATR 고정)
    def _eval_variant(df_full_var, ref_b, pred_b, hist_start):
        df_roll = df_full_static[df_full_static["timestamp"] >= (ROLL_START - pd.Timedelta(hours=72))].reset_index(drop=True)
        df_hist = df_full_var[df_full_var["timestamp"] >= hist_start].reset_index(drop=True)

        cands = get_candidates(df_hist, (ref_b["start"], ref_b["end"]), ex_margin_days=exd, topN=topN, past_only=True)
        if not cands:
            return None

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

        t_entry = pred_b["start"] + delayTD
        if t_entry > pred_b["end"]:
            return None

        pred_seg = df_roll[(df_roll["timestamp"] >= pred_b["start"]) & (df_roll["timestamp"] <= t_entry)]
        if len(pred_seg) == 0:
            return None

        base_cur = float(pred_seg["close"].iloc[0])
        a = ((pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)
        L = len(a)

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

        choose = res_static

        df_roll = choose["df_roll"]
        df_hist = choose["df_hist"]
        best = choose["best"]
        L = choose["L"]
        t_entry = choose["t_entry"]
        pred_seg = choose["pred_seg"]

        hist_full = np.array(best["flow"]["pct"], dtype=float)
        base_now = float(hist_full[L - 1])
        fut = hist_full[L - 1:] - base_now

        idx_max = int(np.argmax(fut)) if fut.size > 0 else 0
        idx_min = int(np.argmin(fut)) if fut.size > 0 else 0
        max_up = float(np.max(fut)) if fut.size > 0 else 0.0
        min_dn = float(np.min(fut)) if fut.size > 0 else 0.0

        ext_start = pred_b["start"] - pd.Timedelta(hours=48)
        prefix_end = min(t_entry, pred_b["end"])
        ext_seg = df_roll[
            (df_roll["timestamp"] >= ext_start) &
            (df_roll["timestamp"] <= prefix_end)
        ].reset_index(drop=True)
        used_ext = (len(ext_seg) >= 2)
        seg = ext_seg if used_ext else pred_seg
        anchor = float(seg["close"].iloc[0]); last = float(seg["close"].iloc[-1])
        ret_pct = (last / anchor - 1.0) * 100.0

        thr_ext  = -1.0
        thr_pred =  0.0
        cutoff   = (thr_ext if used_ext else thr_pred)
        regime_down = (ret_pct < cutoff)
        sim_gate = float(sim_gate_base) + (0.05 if regime_down else 0.0)
        LO_THR_USE = LO_THR + (0.5 if regime_down else 0.0)
        HI_THR_USE = HI_THR + (0.5 if regime_down else 0.0)

        side = "HOLD"
        if best["sim"] >= sim_gate:
            mag_up = abs(max_up); mag_dn = abs(min_dn); mag = max(mag_up, mag_dn)
            if mag >= LO_THR_USE:
                if regime_down and (mag_up >= mag_dn):
                    side = "HOLD"
                else:
                    side = "LONG" if mag_up >= mag_dn else "SHORT"

        entry_time, entry_price, entry_target = (None, None, None)

        if side in ("LONG", "SHORT"):
            if max(abs(max_up), abs(min_dn)) >= HI_THR_USE:
                etime, eprice = make_entry_at(df_roll, t_entry, rule=ENTRY_RULE_FIXED)
                if etime is not None and etime < t_entry:
                    seg_after = df_roll[df_roll["timestamp"] > t_entry]
                    if not seg_after.empty:
                        etime = seg_after["timestamp"].iloc[0]
                        eprice = float(seg_after["open"].iloc[0])
                entry_time, entry_price = etime, eprice
            else:
                df_next_best = best["flow"]["df_next"]
                base_hist_close = float(best["flow"]["base_close"])
                cur_28h_close = _get_close_at_or_next(df_roll, t_entry)

                if cur_28h_close is not None:
                    if side == "LONG":
                        end_k = min((L - 1) + idx_max, len(df_next_best) - 1)
                        lows_slice = df_next_best["low"].iloc[:end_k + 1].to_numpy(dtype=float)
                        if lows_slice.size > 0:
                            low_min = float(np.min(lows_slice))
                            drop_pct = (low_min / base_hist_close - 1.0) * 100.0
                            mag_adj = _adjust_magnitude(abs(drop_pct))
                            entry_target = cur_28h_close * (1.0 + (-mag_adj) / 100.0)
                            entry_time, entry_price = _touch_entry(df_roll, t_entry, pred_b["end"], "LONG", entry_target)
                    else:
                        end_k = min((L - 1) + idx_min, len(df_next_best) - 1)
                        highs_slice = df_next_best["high"].iloc[:end_k + 1].to_numpy(dtype=float)
                        if highs_slice.size > 0:
                            high_max = float(np.max(highs_slice))
                            up_pct = (high_max / base_hist_close - 1.0) * 100.0
                            mag_adj = _adjust_magnitude(abs(up_pct))
                            entry_target = cur_28h_close * (1.0 + mag_adj / 100.0)
                            entry_time, entry_price = _touch_entry(df_roll, t_entry, pred_b["end"], "SHORT", entry_target)

        atr_ref = None
        if entry_time is not None:
            row_at = df_roll[df_roll["timestamp"] == entry_time]
            if not row_at.empty and row_at["atr"].notna().any():
                atr_ref = float(row_at["atr"].fillna(method='ffill').iloc[0])

        sl, tp = (None, None)
        if side in ("LONG", "SHORT") and entry_time is not None and entry_price is not None:
            # ATR 고정
            sl, tp = make_sl_tp(
                entry_price, side, method="ATR", atr=atr_ref,
                sl_pct=None, tp_pct=None, k_sl=k_sl, k_tp=k_tp, tick_size=0.0
            )
        else:
            if side in ("LONG", "SHORT"):
                side = "HOLD"

        size = 0.0
        used_lev = 0.0
        cap_hit = False
        if side in ("LONG", "SHORT") and entry_time is not None and entry_price is not None and sl:
            size = float(eq_run) * float(max_leverage)   # 리스크% 제거: 단순 레버리지 캡 노션널
            used_lev = float(max_leverage)
            cap_hit = False

        exit_time, exit_price, gross_ret, net_ret = (None, None, None, None)
        if side in ("LONG", "SHORT") and entry_time is not None and entry_price is not None:
            exit_time, exit_price, gross_ret, net_ret = simulate_trade(
                df_roll, t_entry, pred_b["end"], side,
                entry_time, entry_price, sl, tp,
                fee_entry=fee_entry, fee_exit=fee_exit,
                slip_entry=slip_entry, slip_exit=slip_exit,
                exit_on_close=True
            )
        else:
            if side in ("LONG", "SHORT"):
                side = "HOLD"

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
            "scaler": "static",
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
            "skip_reason": None,
        })

        pbar.progress(int(100 * (j + 1) / total))

    if not trade_logs:
        st.info("ROLLING 결과 없음")
        st.stop()

    df_log = pd.DataFrame(trade_logs).sort_values("pred_start").reset_index(drop=True)

    df_show = df_log.copy()
    df_show = df_show.drop(columns=["gross_ret_%", "net_ret_%"], errors="ignore")
    df_show = df_show.rename(columns={"ret_equity_%": "ret_%(levered)"})
    cols = [
        "pred_start", "pred_end", "t_entry", "side", "sim_prefix", "scaler",
        "entry_time", "entry", "entry_target", "SL", "TP",
        "size_notional", "used_lev", "cap_hit", "pnl_usd", "ret_%(levered)",
        "eq_before", "eq_after", "exit_time", "exit"
    ]
    df_show = df_show[[c for c in cols if c in df_show.columns]]

    st.markdown("### 결과 테이블 (레버리지 반영 수익률)")
    st.caption("ret_%(levered) = net_ret_% × (size_notional / eq_before)")
    st.dataframe(df_show, use_container_width=True)

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
# NOW (B)
# ---------------------------
elif sim_mode == "NOW-하락/횡보":
    st.subheader("NOW-하락/횡보: 28h 지연 엔트리 · 1회 거래 (2020 풀 기반)")

    SCALE_END_B   = pd.Timestamp("2020-11-01 00:00:00")
    POOL_START_B  = pd.Timestamp("2020-01-01 00:00:00")
    POOL_END_B    = pd.Timestamp("2020-11-01 00:00:00")

    df_full_b = apply_static_zscore(df_feat.copy(), GLOBAL_Z_COLS, SCALE_END_B)
    df_full_b = finalize_preprocessed(df_full_b, window_size)
    pool_df_b = df_full_b[(df_full_b["timestamp"] >= POOL_START_B) & (df_full_b["timestamp"] < POOL_END_B)].reset_index(drop=True)
    if len(pool_df_b) < window_size:
        st.error("NOW-하락/횡보: 2020 풀 데이터가 부족합니다."); st.stop()

    (ref_start, ref_end), (pred_start, pred_end) = pick_blocks(df_full_b["timestamp"].iloc[-1], step_hours=step_hours)

    cands = get_candidates_a(
        df_pool=pool_df_b,
        ref_range=(ref_start, ref_end),
        df_ref=df_full_b,
        feat_cols=FEAT_COLS,
        step_hours=step_hours,
        window_size=window_size,
        sim_mode="DTW", w_dtw=0.5,
        topN=5 if fast else 10,
        ex_margin_days=10 if fast else 5
    )

    results = []
    stepTD = pd.Timedelta(hours=step_hours)
    for f in cands:
        next_start = f["end"]
        next_end   = next_start + stepTD
        df_next = pool_df_b[(pool_df_b["timestamp"] >= next_start) & (pool_df_b["timestamp"] < next_end)]
        if len(df_next) < window_size:
            continue
        closes = df_next["close"].to_numpy()
        base   = float(df_next["open"].iloc[0])
        pct_raw = (closes - base) / base * 100.0
        ENTRY_DELAY_BARS = max(1, int(np.ceil(ENTRY_DELAY_HOURS / 4)))  # 28h -> 7 bars
        base_close_28h = float(df_next["close"].iloc[ENTRY_DELAY_BARS - 1]) \
            if len(df_next) >= ENTRY_DELAY_BARS else float(df_next["close"].iloc[-1])
        results.append({
            "sim": f["sim"],
            "next_start": next_start,
            "next_end": next_end,
            "pct": pct_raw,
            "df_next": df_next.reset_index(drop=True),
            "base_close": base,
            "base_close_28h": base_close_28h
        })

    now_ts_b = df_full_b["timestamp"].iloc[-1]
    t_entry = pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS)
    if now_ts_b < t_entry:
        st.info(f"데이터 부족: 엔트리 고려 시점({t_entry})까지 28h가 지나지 않음.")
        st.stop()

    cur_pred_seg = df_full_b[
        (df_full_b["timestamp"] >= pred_start) &
        (df_full_b["timestamp"] <= min(now_ts_b, pred_end))
    ]
    if len(cur_pred_seg) == 0 or len(results) == 0:
        st.info("데이터 부족")
        st.stop()

    base_cur = float(cur_pred_seg["open"].iloc[0])
    a_plot = ((cur_pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)

    prefix_end = min(t_entry, pred_end)
    cur_prefix = cur_pred_seg[cur_pred_seg["timestamp"] <= prefix_end]
    a = ((cur_prefix["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)
    ENTRY_DELAY_BARS = int(np.ceil(ENTRY_DELAY_HOURS / 4.0))  # 28h -> 7
    L_use = ENTRY_DELAY_BARS + 1
    a = a[:min(L_use, len(a))]
    L = len(a)

    best = None
    for r in results:
        b = np.array(r["pct"], dtype=float)[:L]
        sim_shape = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) else float(cosine_similarity([a], [b])[0][0])
        if (best is None) or (sim_shape > best["sim"]):
            best = {"sim": sim_shape, "flow": r}

    df_best_next = best["flow"]["df_next"]
    base_hist_close = best["flow"].get(
        "base_close_28h",
        best["flow"].get(
            "base_close",
            (float(df_best_next["close"].iloc[L-1]) if len(df_best_next) >= L and L>0 else float(df_best_next["close"].iloc[-1]))
        )
    )

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

    fig, ax = plt.subplots(figsize=(9, 3))
    hist_full = np.array(best["flow"]["pct"], dtype=float)
    ax.plot(np.arange(len(hist_full)), hist_full, label="매칭 72h(원시%)")
    ax.plot(np.arange(len(a_plot)), a_plot, label=f"현재 진행 (L={len(a_plot)})")
    ax.axvline(L - 1, ls="--", label="엔트리 기준(28h)")
    ax.axhline(HI_THR, ls="--"); ax.axhline(-HI_THR, ls="--")
    ax.axhline(LO_THR, ls=":"); ax.axhline(-LO_THR, ls=":")
    ax.axhline(0, ls=":"); ax.set_title("NOW-하락/횡보: 28h 기준 · 진행 vs 매칭 (원시%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.caption("참고: 2020 풀 기준. 28h(세로 점선) 이후의 후보 후행분포에서 상승/하락 우위를 판정합니다.")
    st.pyplot(fig)

    fut = hist_full[min(int(np.ceil(ENTRY_DELAY_HOURS/4.0)), len(hist_full) - 1):] - hist_full[min(int(np.ceil(ENTRY_DELAY_HOURS/4.0)), len(hist_full) - 1)]
    idx_max = int(np.argmax(fut)) if fut.size > 0 else 0
    idx_min = int(np.argmin(fut)) if fut.size > 0 else 0
    max_up = float(np.max(fut)) if fut.size > 0 else 0.0
    min_dn = float(np.min(fut)) if fut.size > 0 else 0.0

    ext_start = pred_start - pd.Timedelta(hours=48)
    prefix_end = min(t_entry, pred_end)
    ext_seg = df_full_b[
        (df_full_b["timestamp"] >= ext_start) &
        (df_full_b["timestamp"] <= prefix_end)
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
    sim_gate = float(sim_gate_base) + (0.05 if regime_down else 0.0)
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

    STRAT_DESC = {
        "A": "강한 상승: HI_THR_USE 이상 상승 우위 → 다음봉 시가 진입",
        "B": "강한 하락: HI_THR_USE 이상 하락 우위 → 다음봉 시가 진입",
        "C": "중간 상승: LO~HI, 상승 우위 → 되돌림 리밋가 (가정값)",
        "C′": "중간 하락: LO~HI, 하락 우위 → 되돌림 리밋가 (가정값)",
        "D": "항상 HOLD",
        "E": "약함/미달 → HOLD"
    }

    def compute_limit_target_local(side: str, df_next_best: pd.DataFrame,
                                   L_local: int, idx_max_local: int, idx_min_local: int,
                                   cur_28h_close_local: float, base_hist_close_28h_local: float):
        if side == "LONG":
            end_k = min((L_local - 1) + idx_max_local, len(df_next_best) - 1)
            lows_slice = df_next_best["low"].iloc[:end_k + 1].to_numpy(dtype=float)
            if lows_slice.size == 0: return None
            low_min = float(np.min(lows_slice))
            drop_pct = (low_min / base_hist_close_28h_local - 1.0) * 100.0
            mag_adj = _adjust_magnitude(abs(drop_pct))
            return cur_28h_close_local * (1.0 + (-mag_adj) / 100.0)
        else:
            end_k = min((L_local - 1) + idx_min_local, len(df_next_best) - 1)
            highs_slice = df_next_best["high"].iloc[:end_k + 1].to_numpy(dtype=float)
            if highs_slice.size == 0: return None
            high_max = float(np.max(highs_slice))
            up_pct = (high_max / base_hist_close_28h_local - 1.0) * 100.0
            mag_adj = _adjust_magnitude(abs(up_pct))
            return cur_28h_close_local * (1.0 + mag_adj / 100.0)

    def next_open_after_local(ts):
        seg_after = df_full_b[df_full_b["timestamp"] > ts]
        return (seg_after["timestamp"].iloc[0], float(seg_after["open"].iloc[0])) if not seg_after.empty else (None, None)

    force_mode = st.checkbox(
        "강제 가정으로 계산(조건 불충족이어도 값 채움)",
        value=True,
        help="2020 풀 기반 C/C′ 리밋 미체결 시에도 가정값으로 진입가를 채워 SL/TP를 계산해 비교합니다."
    )

    cur_28h_close = _get_close_at_or_next(df_full_b, t_entry)
    base_hist_close_local = base_hist_close

    def scenario_row_now(tag: str, plan_side: str, cond_ok: bool, force_mode_local: bool):
        note = ""
        if tag in ("D", "E"):
            return {"scenario": tag, "설명": STRAT_DESC.get(tag, ""), "side": "HOLD", "t_entry": t_entry,
                    "entry_time": None, "entry_price": None, "SL": None, "TP": None,
                    "cond_ok": cond_ok, "touched": False, "forced": False, "used_rule": "HOLD", "note": "항상 HOLD"}

        if tag in ("A", "B"):
            et0, ep0 = make_entry_at(df_full_b, t_entry, rule=ENTRY_RULE_FIXED)
            if et0 is not None and et0 < t_entry:
                et0, ep0 = next_open_after_local(t_entry)
            entry_time, entry_price = et0, ep0
            touched = False; forced = False; used_rule = "다음봉 시가"
        else:
            entry_time = None; entry_price = None; touched = False; forced = False; used_rule = "리밋가(가정)"
            if cur_28h_close is not None and len(df_best_next) > 0:
                if tag == "C":
                    target = compute_limit_target_local("LONG", df_best_next, L, idx_max, idx_min, cur_28h_close, base_hist_close_local)
                    if target is not None:
                        et, ep = _touch_entry(df_full_b, t_entry, pred_end, "LONG", target)
                        if et is not None: entry_time, entry_price, touched = et, ep, True
                        elif force_mode_local: entry_time, entry_price, forced = t_entry, float(target), True
                else:
                    target = compute_limit_target_local("SHORT", df_best_next, L, idx_max, idx_min, cur_28h_close, base_hist_close_local)
                    if target is not None:
                        et, ep = _touch_entry(df_full_b, t_entry, pred_end, "SHORT", target)
                        if et is not None: entry_time, entry_price, touched = et, ep, True
                        elif force_mode_local: entry_time, entry_price, forced = t_entry, float(target), True
            if (entry_price is None) and force_mode_local:
                et0, ep0 = make_entry_at(df_full_b, t_entry, rule=ENTRY_RULE_FIXED)
                if et0 is not None and et0 < t_entry: et0, ep0 = next_open_after_local(t_entry)
                entry_time, entry_price = et0, ep0; note += "리밋 불가→시가 대체; "

        SL = TP = None
        side_out = plan_side if tag in ("C", "C′") else ("LONG" if tag == "A" else "SHORT")
        if entry_time is not None and entry_price is not None:
            row_at = df_full_b[df_full_b["timestamp"] == entry_time]
            atr_ref_local = float(row_at["atr"].fillna(method='ffill').iloc[0]) if (not row_at.empty and row_at["atr"].notna().any()) else None
            # ATR 고정 (sl_pct/tp_pct 사용 안함)
            SL, TP = make_sl_tp(entry_price, side_out, method="ATR", atr=atr_ref_local,
                                 sl_pct=None, tp_pct=None, k_sl=k_sl, k_tp=k_tp, tick_size=0.0)
        else:
            side_out = "HOLD"

        return {"scenario": tag, "설명": STRAT_DESC.get(tag, ""), "side": side_out, "t_entry": t_entry,
                "entry_time": entry_time, "entry_price": entry_price, "SL": SL, "TP": TP,
                "cond_ok": cond_ok, "touched": touched, "forced": forced, "used_rule": used_rule, "note": note}

    rows = []
    cond_A = (mag >= HI_THR_USE) and up_win and (not regime_down)
    rows.append(scenario_row_now("A", "LONG", cond_A, force_mode))
    cond_B = (mag >= HI_THR_USE) and dn_win
    rows.append(scenario_row_now("B", "SHORT", cond_B, force_mode))
    cond_C = (LO_THR_USE <= mag < HI_THR_USE) and up_win and (not regime_down)
    rows.append(scenario_row_now("C", "LONG", cond_C, force_mode))
    cond_Cp = (LO_THR_USE <= mag < HI_THR_USE) and dn_win
    rows.append(scenario_row_now("C′", "SHORT", cond_Cp, force_mode))
    cond_D = (LO_THR_USE <= mag < HI_THR_USE) and up_win and regime_down
    rows.append(scenario_row_now("D", "HOLD", cond_D, force_mode))
    cond_E = (mag < LO_THR_USE) or (best["sim"] < sim_gate)
    rows.append(scenario_row_now("E", "HOLD", cond_E, force_mode))

    if st.button(
        "시나리오 계산",
        help="NOW-하락/횡보 규칙으로 A~E 시나리오별 진입/SL/TP, 거리(Δ), %를 표로 계산합니다."
    ):
        df_scn = pd.DataFrame(rows)

        def _delta(row):
            ep = row.get("entry_price"); sl = row.get("SL"); tp = row.get("TP")
            if ep is None or sl is None or tp is None:
                return pd.Series([None, None, None, None])
            sl_d = abs(ep - sl); tp_d = abs(tp - ep)
            sl_pct_v = (sl_d / ep) * 100.0; tp_pct_v = (tp_d / ep) * 100.0
            return pd.Series([sl_d, tp_d, sl_pct_v, tp_pct_v])

        df_scn[["SL_Δ","TP_Δ","SL_%","TP_%"]] = df_scn.apply(_delta, axis=1)
        show_cols = ["scenario","설명","side","t_entry","entry_time","entry_price","used_rule","cond_ok","touched","forced","SL","TP","SL_Δ","TP_Δ","SL_%","TP_%","note"]
        df_scn = df_scn[[c for c in show_cols if c in df_scn.columns]]
        st.dataframe(df_scn, use_container_width=True)

# ---------------------------
# ROLLING (B) — FIXED 제거 & fast 강제 True
# ---------------------------
elif sim_mode == "BT-하락/횡보":
    st.subheader("BT-하락/횡보: 28h 지연 엔트리 · 블록당 1회 거래 백테스트 (2020 풀 기반, ATR 고정, fast 모드)")

    col1, col2, col3 = st.columns(3)

    with col1:
        sim_engine = st.selectbox(
            "유사도 방식",
            ["DTW", "Cosine"],
            index=0,
            help="과거 구간과의 유사도 계산 메트릭. DTW 또는 Cosine만 허용."
        )

    with col2:
        k_sl = st.number_input(
            "k_sl(×ATR)",
            min_value=0.1, max_value=10.0, value=1.5, step=0.1,
            help="손절폭 = k_sl × ATR. 예) 1.5면 엔트리에서 1.5×ATR 반대방향."
        )
        k_tp = st.number_input(
            "k_tp(×ATR)",
            min_value=0.1, max_value=20.0, value=2.5, step=0.1,
            help="익절폭 = k_tp × ATR. 예) 3.0면 엔트리에서 3×ATR 유리한 방향."
        )

        fee_entry  = st.number_input(
            "Entry fee %",
            min_value=0.0, max_value=1.0, value=0.04, step=0.01,
            help="진입 수수료(%). 백테스트 체결 현실화 가정."
        ) / 100.0
        fee_exit   = st.number_input(
            "Exit fee %",
            min_value=0.0, max_value=1.0, value=0.05, step=0.01,
            help="청산(종료) 수수료(%)."
        ) / 100.0
        slip_entry = st.number_input(
            "Slippage entry %",
            min_value=0.0, max_value=1.0, value=0.03, step=0.01,
            help="진입 시 슬리피지 가정(%)."
        ) / 100.0
        slip_exit  = st.number_input(
            "Slippage exit %",
            min_value=0.0, max_value=1.0, value=0.05, step=0.01,
            help="청산 시 슬리피지 가정(%)."
        ) / 100.0

    with col3:
        equity = st.number_input(
            "가상 Equity (USDT)",
            min_value=10.0,
            value=(float(equity) if equity is not None else 1000.0),
            step=10.0,
            help="백테스트/포지션 사이징에 사용하는 가상의 계정 잔고(USDT)."
        )
        max_leverage = st.number_input(
            "최대 레버리지(x)",
            min_value=1.0, max_value=50.0,
            value=(float(max_leverage) if max_leverage is not None else 10.0),
            step=1.0,
            help="사이징 계산 시 사용할 레버리지 상한(실체결 한도 아님)."
        )

    sim_gate_base = 0.75
    topN = 5              # fast 강제
    ex_margin_days = 10   # fast 강제
    ROLL_START_B = pd.Timestamp("2025-01-01 00:00:00")
    step_hours = 72
    window_size = 18
    ENTRY_DELAY_HOURS = 28
    stepTD  = pd.Timedelta(hours=step_hours)
    delayTD = pd.Timedelta(hours=ENTRY_DELAY_HOURS)

    SCALE_END_B  = pd.Timestamp("2020-11-01 00:00:00")
    POOL_START_B = pd.Timestamp("2020-01-01 00:00:00")
    POOL_END_B   = pd.Timestamp("2020-11-01 00:00:00")

    df_full_b = apply_static_zscore(df_feat.copy(), GLOBAL_Z_COLS, SCALE_END_B)
    df_full_b = finalize_preprocessed(df_full_b, window_size)

    pool_df_b = df_full_b[
        (df_full_b["timestamp"] >= POOL_START_B) &
        (df_full_b["timestamp"] <  POOL_END_B)
    ].reset_index(drop=True)

    df_roll_base = df_full_b[df_full_b["timestamp"] >= (ROLL_START_B - pd.Timedelta(hours=72))].reset_index(drop=True)
    if len(df_roll_base) < window_size:
        st.warning("BT-하락/횡보: 데이터 부족"); st.stop()

    blocks_all = enumerate_blocks(df_roll_base, step_hours=step_hours, window_size=window_size)
    start_idx = None
    for i in range(1, len(blocks_all)):
        if blocks_all[i]["start"] >= ROLL_START_B:
            start_idx = i; break
    if start_idx is None:
        st.warning("BT-하락/횡보: 시작 이후 pred 블록 없음"); st.stop()

    trade_logs = []
    pbar = st.progress(0)
    total = (len(blocks_all) - start_idx)
    eq_run = float(equity)

    for j, bp in enumerate(range(start_idx, len(blocks_all))):
        ref_b  = blocks_all[bp - 1]
        pred_b = blocks_all[bp]
        t_entry = pred_b["start"] + delayTD
        if t_entry > pred_b["end"]:
            pbar.progress(int(100 * (j + 1) / max(1, total))); continue

        cands = get_candidates_a(
            df_pool=pool_df_b,
            ref_range=(ref_b["start"], ref_b["end"]),
            df_ref=df_full_b,
            feat_cols=FEAT_COLS,
            step_hours=step_hours, window_size=window_size,
            sim_mode=sim_engine, w_dtw=0.5,  # w_dtw는 무시됨 (Hybrid 없음)
            topN=topN, ex_margin_days=ex_margin_days
        )
        if not cands:
            pbar.progress(int(100 * (j + 1) / max(1, total))); continue

        results = []
        for f in cands:
            next_start = f["end"]; next_end = next_start + stepTD
            df_next = pool_df_b[(pool_df_b["timestamp"] >= next_start) & (pool_df_b["timestamp"] < next_end)]
            if len(df_next) < window_size:
                continue
            closes = df_next["close"].to_numpy()
            baseC  = float(closes[0])
            pct_c  = (closes - baseC) / baseC * 100.0
            results.append({
                "sim": f["sim"], "next_start": next_start, "next_end": next_end,
                "pct": pct_c, "df_next": df_next.reset_index(drop=True), "base_close": baseC
            })
        if not results:
            pbar.progress(int(100 * (j + 1) / max(1, total))); continue

        pred_seg = df_roll_base[(df_roll_base["timestamp"] >= pred_b["start"]) & (df_roll_base["timestamp"] <= t_entry)]
        if len(pred_seg) == 0:
            pbar.progress(int(100 * (j + 1) / max(1, total))); continue

        base_cur = float(pred_seg["close"].iloc[0])
        a = ((pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)
        L = len(a)

        best = None
        for r in results:
            b = np.array(r["pct"], dtype=float)[:L]
            sim_shape = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) else float(cosine_similarity([a],[b])[0][0])
            if (best is None) or (sim_shape > best["sim"]):
                best = {"sim": sim_shape, "flow": r}

        hist_full = np.array(best["flow"]["pct"], dtype=float)
        base_now  = float(hist_full[L - 1]) if L > 0 else 0.0
        fut       = hist_full[L - 1:] - base_now
        idx_max   = int(np.argmax(fut)) if fut.size > 0 else 0
        idx_min   = int(np.argmin(fut)) if fut.size > 0 else 0
        max_up    = float(np.max(fut))  if fut.size > 0 else 0.0
        min_dn    = float(np.min(fut))  if fut.size > 0 else 0.0

        ext_start = pred_b["start"] - pd.Timedelta(hours=48)
        prefix_end = min(t_entry, pred_b["end"])
        ext_seg = df_roll_base[(df_roll_base["timestamp"] >= ext_start) & (df_roll_base["timestamp"] <= prefix_end)].reset_index(drop=True)
        used_ext = (len(ext_seg) >= 2)
        seg = ext_seg if used_ext else pred_seg
        anchor = float(seg["close"].iloc[0]); last = float(seg["close"].iloc[-1])
        ret_pct = (last / anchor - 1.0) * 100.0
        cutoff  = (-1.0 if used_ext else 0.0)
        regime_down = (ret_pct < cutoff)

        sim_gate    = float(sim_gate_base) + (0.05 if regime_down else 0.0)
        LO_THR_USE  = LO_THR + (0.5 if regime_down else 0.0)
        HI_THR_USE  = HI_THR + (0.5 if regime_down else 0.0)

        side = "HOLD"
        if best["sim"] >= sim_gate:
            mag_up = abs(max_up); mag_dn = abs(min_dn); mag = max(mag_up, mag_dn)
            if mag >= LO_THR_USE:
                if regime_down and (mag_up >= mag_dn):
                    side = "HOLD"
                else:
                    side = "LONG" if mag_up >= mag_dn else "SHORT"

        entry_time = entry_price = entry_target = None
        if side in ("LONG","SHORT"):
            if max(abs(max_up), abs(min_dn)) >= HI_THR_USE:
                etime, eprice = make_entry_at(df_roll_base, t_entry, rule="다음봉 시가")
                if etime is not None and etime < t_entry:
                    seg_after = df_roll_base[df_roll_base["timestamp"] > t_entry]
                    if not seg_after.empty:
                        etime = seg_after["timestamp"].iloc[0]
                        eprice = float(seg_after["open"].iloc[0])
                entry_time, entry_price = etime, eprice
            else:
                df_next_best = best["flow"]["df_next"]
                base_hist_close = float(best["flow"]["base_close"])
                cur_28h_close = _get_close_at_or_next(df_roll_base, t_entry)
                if (cur_28h_close is not None) and (len(df_next_best) > 0):
                    if side == "LONG":
                        end_k = min((L - 1) + idx_max, len(df_next_best) - 1)
                        lows_slice = df_next_best["low"].iloc[:end_k + 1].to_numpy(dtype=float)
                        if lows_slice.size > 0:
                            low_min = float(np.min(lows_slice))
                            drop_pct = (low_min / base_hist_close - 1.0) * 100.0
                            mag_adj = _adjust_magnitude(abs(drop_pct))
                            entry_target = cur_28h_close * (1.0 - mag_adj/100.0)
                            entry_time, entry_price = _touch_entry(df_roll_base, t_entry, pred_b["end"], "LONG", entry_target)
                    else:
                        end_k = min((L - 1) + idx_min, len(df_next_best) - 1)
                        highs_slice = df_next_best["high"].iloc[:end_k + 1].to_numpy(dtype=float)
                        if highs_slice.size > 0:
                            high_max = float(np.max(highs_slice))
                            up_pct = (high_max / base_hist_close - 1.0) * 100.0
                            mag_adj = _adjust_magnitude(abs(up_pct))
                            entry_target = cur_28h_close * (1.0 + mag_adj/100.0)
                            entry_time, entry_price = _touch_entry(df_roll_base, t_entry, pred_b["end"], "SHORT", entry_target)

        atr_ref = None
        if entry_time is not None:
            row_at = df_roll_base[df_roll_base["timestamp"] == entry_time]
            if not row_at.empty and row_at["atr"].notna().any():
                atr_ref = float(row_at["atr"].fillna(method='ffill').iloc[0])

        SL = TP = None
        if side in ("LONG","SHORT") and (entry_time is not None) and (entry_price is not None):
            SL, TP = make_sl_tp(
                entry_price, side,
                method="ATR",   # ATR 고정
                atr=atr_ref, sl_pct=None, tp_pct=None, k_sl=k_sl, k_tp=k_tp, tick_size=0.0
            )
        else:
            side = "HOLD"

        size = used_lev = 0.0; cap_hit = False
        exit_time = exit_price = gross_ret = net_ret = None
        if side in ("LONG","SHORT") and (entry_time is not None) and (entry_price is not None) and (SL is not None):
            size = float(eq_run) * float(max_leverage)   # 리스크% 제거: 단순 레버리지 캡 노션널
            used_lev = float(max_leverage)
            cap_hit = False
            exit_time, exit_price, gross_ret, net_ret = simulate_trade(
                df=df_roll_base, start_ts=pred_b["start"], end_ts=pred_b["end"], side=side,
                entry_time=entry_time, entry_price=float(entry_price),
                sl=SL, tp=TP,
                fee_entry=fee_entry, fee_exit=fee_exit,
                slip_entry=slip_entry, slip_exit=slip_exit,
                exit_on_close=True
            )
        else:
            side = "HOLD"

        ret_pct_trade = (net_ret or 0.0) / 100.0
        eq_before = eq_run
        pnl_usd = (size or 0.0) * ret_pct_trade
        eq_run = eq_run + pnl_usd
        ret_equity_pct = (pnl_usd / (eq_before if eq_before > 0 else 1.0)) * 100.0

        trade_logs.append({
            "pred_start": pred_b["start"], "pred_end": pred_b["end"], "t_entry": t_entry,
            "side": side, "sim_prefix": best["sim"],
            "entry_time": entry_time, "entry": entry_price, "entry_target": entry_target,
            "SL": SL, "TP": TP,
            "size_notional": size, "used_lev": used_lev, "cap_hit": cap_hit,
            "exit_time": exit_time, "exit": exit_price,
            "gross_ret_%": gross_ret, "net_ret_%": net_ret,
            "eq_before": eq_before, "eq_after": eq_run, "pnl_usd": pnl_usd, "ret_equity_%": ret_equity_pct
        })

        pbar.progress(int(100 * (j + 1) / max(1, total)))

    if not trade_logs:
        st.info("BT-하락/횡보 결과 없음"); st.stop()

    df_log = pd.DataFrame(trade_logs).sort_values("pred_start").reset_index(drop=True)

    df_show = (df_log.copy()
               .drop(columns=["gross_ret_%","net_ret_%"], errors="ignore")
               .rename(columns={"ret_equity_%": "ret_%(levered)"}))
    cols = ["pred_start","pred_end","t_entry","side","entry_time","entry","entry_target",
            "SL","TP","size_notional","used_lev","cap_hit","pnl_usd","ret_%(levered)",
            "eq_before","eq_after","exit_time","exit","sim_prefix"]
    df_show = df_show[[c for c in cols if c in df_show.columns]]

    st.markdown("### 결과 테이블 (레버리지 반영 수익률) — B")
    st.caption("ret_%(levered) = net_ret_% × (size_notional / eq_before)")
    st.dataframe(df_show, use_container_width=True)

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
        ax.plot(dates, equity_curve, linewidth=2, label="Equity (B)")
        ax.set_title("Equity Curve (net) — ROLLING (B)")
        ax.grid(True, alpha=0.3); ax.legend()
        st.pyplot(fig)
    else:
        st.warning("에쿼티 커브를 그릴 수 없습니다.")
