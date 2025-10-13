# === Prelude: 한글 폰트/음수 디폴트 적용 (UI 없음) ===
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False      # 음수 기호 깨짐 방지
# === End Prelude ===

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from connectors import (
    connect_binance, connect_binance_trade,
    get_futures_balances, get_futures_positions,
    ensure_leverage_and_margin, get_symbol_filters,
)
from data_fetch import fetch_futures_4h_klines, fetch_funding_rate
import features as F
from features import (
    add_features, apply_static_zscore, finalize_preprocessed,
    window_is_finite, window_vector, GLOBAL_Z_COLS, align_external_to_klines
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
from news_fetch import build_news_frame

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

    # SL/TP은 기본 ATR 사용.
    sltp_method = "ATR"
    k_sl = 1.0
    k_tp = 2.5

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

df_news = build_news_frame(start_ts="2020-01-01", end_ts=None)

def _attach_news_feature(df_feat_local: pd.DataFrame, df_news_local: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    df_feat(베이스)와 df_news(외부) timestamp의 dtype을 맞춰서 merge_asof가 터지지 않게 한다.
    - 베이스가 tz-aware면 ext도 tz-aware(UTC)로 맞추고, 둘 다 정렬한다.
    - 베이스가 tz-naive면 ext도 tz-naive(UTC 기준 값에서 tz 제거)로 맞춘다.
    - df_news에 해당 컬럼이 없거나 값이 비어도 0.0 stub으로 안전하게 붙인다.
    """
    if (df_news_local is None) or df_news_local.empty:
        # df_news가 비어도 스텁 0.0 컬럼을 만들어 두되, align은 생략
        if col not in df_feat_local.columns:
            df_feat_local = df_feat_local.copy()
            df_feat_local[col] = 0.0
        return df_feat_local

    # 외부 프레임 복사 및 필요한 컬럼만 준비
    ext = df_news_local.copy()
    if "timestamp" not in ext.columns:
        # timestamp 없으면 전부 0 스텁
        if col not in df_feat_local.columns:
            df_feat_local = df_feat_local.copy()
            df_feat_local[col] = 0.0
        return df_feat_local

    # 대상 컬럼이 없으면 0 스텁 만들어 병합해도 무방
    if col not in ext.columns:
        ext[col] = 0.0

    # dtype 정리
    if not is_datetime64_any_dtype(df_feat_local["timestamp"]):
        df_feat_local["timestamp"] = pd.to_datetime(df_feat_local["timestamp"], errors="coerce")
    if not is_datetime64_any_dtype(ext["timestamp"]):
        ext["timestamp"] = pd.to_datetime(ext["timestamp"], errors="coerce")

    base_is_tz = is_datetime64tz_dtype(df_feat_local["timestamp"].dtype)
    ext_is_tz  = is_datetime64tz_dtype(ext["timestamp"].dtype)

    if base_is_tz:
        # 베이스가 tz-aware → ext도 tz-aware(UTC)
        ext["timestamp"] = pd.to_datetime(ext["timestamp"], utc=True)
        # 혹시 베이스에 tz-naive가 섞여 있으면 UTC로 강제 캐스팅
        df_feat_local = df_feat_local.copy()
        df_feat_local["timestamp"] = pd.to_datetime(df_feat_local["timestamp"], utc=True)
    else:
        # 베이스가 tz-naive → ext도 tz-naive(UTC 기준값에서 tz 제거)
        ext["timestamp"] = pd.to_datetime(ext["timestamp"], utc=True).dt.tz_localize(None)
        if is_datetime64tz_dtype(df_feat_local["timestamp"].dtype):
            df_feat_local = df_feat_local.copy()
            df_feat_local["timestamp"] = pd.to_datetime(df_feat_local["timestamp"], utc=True).dt.tz_localize(None)

    # 정렬 필수
    ext = ext.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df_feat_local = df_feat_local.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # 시계열 안전 병합 (features.align_external_to_klines 사용)
    return align_external_to_klines(df_feat_local, ext, col)

# ② news_vol 파생: news_count + cpanic_cnt (둘 중 없으면 0으로)
if (df_news is not None) and (not df_news.empty):
    if "news_count" not in df_news.columns:
        df_news["news_count"] = 0.0
    if "cpanic_cnt" not in df_news.columns:
        df_news["cpanic_cnt"] = 0.0
    df_news["news_vol"] = df_news["news_count"].fillna(0.0) + df_news["cpanic_cnt"].fillna(0.0)

# ③ 꼭 쓸 2개만 부착 (없어도 0으로 들어가게 설계)
df_feat = _attach_news_feature(df_feat, df_news, "news_tone")
df_feat = _attach_news_feature(df_feat, df_news, "news_vol")
# (선택) Google Trends까지 쓰려면 아래 라인 활성화
# df_feat = _attach_news_feature(df_feat, df_news, "trends_btc")

# ④ 정규화·특징 컬럼 등록 (중복 방지)
def _safe_add(lst, item):
    if item not in lst:
        lst.append(item)

_safe_add(GLOBAL_Z_COLS, "news_tone")
_safe_add(GLOBAL_Z_COLS, "news_vol")
_safe_add(F.FEAT_COLS, "news_tone_z")
_safe_add(F.FEAT_COLS, "news_vol_z")

# ⑤ 정규화(훈련 구간 고정) 후 기존 파이프라인 합류
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
    use_cols = [c for c in cols if c in df_window.columns]
    if not use_cols:
        return False
    arr = df_window[use_cols].to_numpy()
    return np.isfinite(arr).all()

def _window_vector_a(df_window, feat_cols, L=18):
    use_cols = [c for c in feat_cols if c in df_window.columns]
    if not use_cols:
        return np.array([], dtype=float)

    X = df_window[use_cols].to_numpy(dtype=float)

    MINMAX_COLS = ['log_ret','atr_z','vol_pct_z']
    for c in MINMAX_COLS:
        if c in use_cols:
            j = use_cols.index(c)
            v = X[:, j]
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            X[:, j] = 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.reshape(-1)

def get_candidates_a(df_pool, ref_range, df_ref, feat_cols, step_hours=72, window_size=18,
                     sim_mode='DTW', w_dtw=0.5, topN=10, ex_margin_days=5):
    # ❗ df_pool/df_ref 모두에 존재하는 피처만 사용
    feat_cols = [c for c in feat_cols if (c in df_pool.columns and c in df_ref.columns)]
    if not feat_cols:
        return []

    ref_seg = df_ref[(df_ref["timestamp"] >= ref_range[0]) & (df_ref["timestamp"] < ref_range[1])]
    if len(ref_seg) < window_size: return []
    wL = ref_seg.iloc[:window_size]
    if not _window_is_finite_a(wL, feat_cols): return []
    vec_ref = _window_vector_a(wL, feat_cols, L=window_size)

    blocks = enumerate_blocks(df_pool, step_hours=step_hours, window_size=window_size)
    ex_margin = pd.Timedelta(days=ex_margin_days)
    F = len(feat_cols)
    cand = []
    for b in blocks:
        if not (b["end"] <= ref_range[0] - ex_margin or b["start"] >= ref_range[1] + ex_margin):
            continue
        w = df_pool[(df_pool["timestamp"] >= b["start"]) & (df_pool["timestamp"] < b["end"])]
        if len(w) < window_size:
            continue
        wL2 = w.iloc[:window_size]
        if not _window_is_finite_a(wL2, feat_cols):
            continue
        vec_hist = _window_vector_a(wL2, feat_cols, L=window_size)
        if vec_hist.size == 0 or vec_ref.size == 0:
            continue
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

    # --- 방어적 F 계산: vec_ref 길이로부터 실제 feature 개수 F 역산 ---
    if len(vec_ref) % window_size != 0:
        # 벡터 길이가 L의 배수가 아니면 스킵(스키마 불일치)
        return []
    F = len(vec_ref) // window_size

    blocks = enumerate_blocks(df, step_hours=step_hours, window_size=window_size)
    ex_margin = pd.Timedelta(days=ex_margin_days)
    cand = []
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

        # 후보 벡터가 (L * F) 길이와 일치하지 않으면 스킵
        if len(vec_hist) != window_size * F:
            continue

        sim = sim_tier3(vec_ref, vec_hist, L=window_size, F=F, mode=sim_engine, w_dtw=w_dtw)
        cand.append({"start": b["start"], "end": b["end"], "sim": sim})
    cand.sort(key=lambda x: x["sim"], reverse=True)
    return cand[:topN]

def _adjust_magnitude(pct_mag: float) -> float:
    return max(0.0, pct_mag-0.1)

def _get_close_at_or_before(df: pd.DataFrame, ts: pd.Timestamp):
    """[정확] ts가 봉 오픈타임이라면 '직전 봉 종가'를 반환. 없으면 가능한 합리적 fallback."""
    if df.empty:
        return None
    exact = df[df["timestamp"] == ts]
    if not exact.empty:
        idx = exact.index[0]
        if idx == 0:
            return float(exact["open"].iloc[0])
        return float(df.iloc[idx - 1]["close"])
    before = df[df["timestamp"] < ts]
    if not before.empty:
        return float(before.iloc[-1]["close"])
    return float(df.iloc[0]["open"])

def _touch_entry(df: pd.DataFrame, start_ts, end_ts, side: str, target_price: float):
    seg = df[(df["timestamp"] > start_ts) & (df["timestamp"] < end_ts)]
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
    cands = get_candidates_a(
    df_pool=df_full,
    ref_range=(ref_start, ref_end),
    df_ref=df_full,
    feat_cols=F.FEAT_COLS,   # features 모듈에서 in-place로 확장한 FEAT_COLS 사용
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
        next_end = next_start + stepTD
        df_next = df_full[(df_full["timestamp"] >= next_start) & (df_full["timestamp"] < next_end)]
        if len(df_next) < window_size:
            continue
        closes = df_next["close"].to_numpy()
        base = float(df_next["open"].iloc[0])  # 0h open (분모로 사용할 값)
        pct_raw = (closes - base) / base * 100.0
        # 28h 종가(없으면 마지막 종가) — 기록은 하되 분모로 쓰지 않음
        ENTRY_DELAY_BARS = max(1, int(np.ceil(ENTRY_DELAY_HOURS / 4)))  # 28h -> 7 bars
        base_close_28h = float(df_next["close"].iloc[ENTRY_DELAY_BARS - 1]) \
            if len(df_next) >= ENTRY_DELAY_BARS else float(df_next["close"].iloc[-1])
        results.append({
            "sim": f["sim"],
            "next_start": next_start,
            "next_end": next_end,
            "pct": pct_raw,
            "df_next": df_next.reset_index(drop=True),
            "base_close": base,             # 0h open 저장
            "base_close_28h": base_close_28h
        })

    # 현재 진행 퍼센트 시계열
    cur_pred_seg = df_full[
        (df_full["timestamp"] >= pred_start) &
        (df_full["timestamp"] <= min(now_ts, pred_end))
    ]
    if len(cur_pred_seg) == 0 or len(results) == 0:
        st.info("데이터 부족")
        st.stop()

    base_cur = float(cur_pred_seg["open"].iloc[0])
    a_plot = ((cur_pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)

    # 프리픽스(0~28h 포함) 길이 고정
    ENTRY_DELAY_BARS = int(np.ceil(ENTRY_DELAY_HOURS / 4.0))  # 28h -> 7
    prefix_end = min(pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS), pred_end)
    cur_prefix = cur_pred_seg[cur_pred_seg["timestamp"] <= prefix_end]
    a = ((cur_prefix["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)
    L_use = ENTRY_DELAY_BARS + 1  # 0~7 포함 → 8개
    a = a[:min(L_use, len(a))]
    L = len(a)

    # 프리픽스 최고 후보 선정 (코사인)
    best = None
    for r in results:
        b = np.array(r["pct"], dtype=float)[:L]
        sim_shape = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) else float(cosine_similarity([a], [b])[0][0])
        if (best is None) or (sim_shape > best["sim"]):
            best = {"sim": sim_shape, "flow": r}

    df_best_next = best["flow"]["df_next"]

    # ✅ 분모 기준을 0h open으로 통일 (base_close 우선)
    base_hist_close = best["flow"].get(
        "base_close",  # 0h open
        best["flow"].get(
            "base_close_28h",
            (float(df_best_next["close"].iloc[L-1]) if len(df_best_next) >= L and L > 0
             else float(df_best_next["close"].iloc[-1]))
        )
    )

    # 표 (퍼센트 테이블)
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
        "현재 블록 구간": f"{pred_start} ~ {pred_end}"
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
    prefix_end = min(pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS), pred_end)
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
    sim_gate = float(sim_gate_base)
    LO_THR_USE = LO_THR
    HI_THR_USE = HI_THR

    mag_up = abs(max_up); mag_dn = abs(min_dn); mag = max(mag_up, mag_dn)
    up_win = mag_up >= mag_dn; dn_win = mag_dn > mag_up

    if mag >= HI_THR_USE:
        if dn_win or (up_win and regime_down):
            current_scenario = "B"
        elif up_win and (not regime_down):
            current_scenario = "A"
        else:
            current_scenario = "E"
    elif LO_THR_USE <= mag < HI_THR_USE:
        if dn_win or (up_win and regime_down):
            current_scenario = "C′"
        elif up_win and (not regime_down):
            current_scenario = "C"
        else:
            current_scenario = "E"
    else:
        current_scenario = "E"

    if best["sim"] < sim_gate:
        current_scenario = "E"

    st.markdown(f"### 📌 현재 판정: **{current_scenario} 시나리오**")
    st.caption(f"현재 유사도 = {best['sim']:.3f} / 게이트 = {sim_gate_base:.2f}")
    st.write(f"🕒 현재 데이터 최신 시점: {now_ts}")

    STRAT_DESC = {
        "A": "강한 상승: HI_THR_USE 이상 & (상승우위) & 비하락레짐 → 다음봉 시가 (LONG)",
        "B": "강한 하락: HI_THR_USE 이상 & (하락우위 또는 하락레짐+상승우위) → 다음봉 시가 (SHORT)",
        "C": "중간 상승: LO~HI & (상승우위) & 비하락레짐 → 되돌림 리밋가 (LONG)",
        "C′": "중간 하락: LO~HI & (하락우위 또는 하락레짐+상승우위) → 되돌림 리밋가 (SHORT)",
        "E": "약함/미달(또는 유사도 미달) → HOLD"
    }

    def compute_limit_target_local(side: str,
                                   df_next_best: pd.DataFrame,
                                   L_local: int,
                                   idx_max_local: int,
                                   idx_min_local: int,
                                   cur_28h_close_local: float,
                                   base_hist_close_local: float):  # 분모: 0h open
        if side == "LONG":
            end_k = min((L_local - 1) + idx_max_local, len(df_next_best) - 1)
            lows_slice = df_next_best["low"].iloc[:end_k + 1].to_numpy(dtype=float)
            if lows_slice.size == 0:
                return None
            low_min = float(np.min(lows_slice))
            drop_pct = (low_min / base_hist_close_local - 1.0) * 100.0   # 0h open 분모
            mag_adj = _adjust_magnitude(abs(drop_pct))
            return cur_28h_close_local * (1.0 + (-mag_adj) / 100.0)
        elif side == "SHORT":
            end_k = min((L_local - 1) + idx_min_local, len(df_next_best) - 1)
            highs_slice = df_next_best["high"].iloc[:end_k + 1].to_numpy(dtype=float)
            if highs_slice.size == 0:
                return None
            high_max = float(np.max(highs_slice))
            up_pct = (high_max / base_hist_close_local - 1.0) * 100.0     # 0h open 분모
            mag_adj = _adjust_magnitude(abs(up_pct))
            return cur_28h_close_local * (1.0 + mag_adj / 100.0)
        return None

    # === 엔트리 기준: 28h 이후 "최소 시작 가능 시점" 고정 ===
    ENTRY_DELAY_BARS = int(np.ceil(ENTRY_DELAY_HOURS / 4.0))  # 28h -> 7 bars
    ENTRY_ANCHOR_TS = pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS)

    _seg_after = df_full[df_full["timestamp"] > ENTRY_ANCHOR_TS]
    if _seg_after.empty:
        ENTRY_FIX_TS, ENTRY_FIX_PRICE = (None, None)
    else:
        ENTRY_FIX_TS  = _seg_after["timestamp"].iloc[0]          # 28h '직후' 첫 오픈
        ENTRY_FIX_PRICE = float(_seg_after["open"].iloc[0])

    # 28h 시점의 "현재 기준 가격"(되돌림 타깃 산출용)
    CUR_28H_CLOSE = _get_close_at_or_before(df_full, ENTRY_ANCHOR_TS)
    if CUR_28H_CLOSE is None and ENTRY_FIX_PRICE is not None:
        CUR_28H_CLOSE = float(ENTRY_FIX_PRICE)

    # 📌 분모로 사용할 히스토리 기준 (0h open)
    base_hist_close_local = float(base_hist_close)

    # --- 시나리오 행 생성 (고정 엔트리 규칙) ---
    def scenario_row_now(tag: str, plan_side: str, cond_ok: bool):
        """
        - A/B: 28h 이후 '첫 봉 시가(ENTRY_FIX_PRICE)'로 고정 진입.
        - C/C′: 되돌림 리밋 타깃을 '항상' 진입가로 고정(터치 여부 무관).
        """
        if tag == "E":
            return {
                "scenario": tag, "설명": STRAT_DESC.get(tag, ""),
                "side": "HOLD","min_entry_time": ENTRY_FIX_TS,
                "entry_price": None, "SL": None, "TP": None,
                "cond_ok": cond_ok, "note": "항상 HOLD"
            }

        # ENTRY_FIX_PRICE 없으면 계산 불가 → HOLD
        if ENTRY_FIX_PRICE is None:
            return {
                "scenario": tag, "설명": STRAT_DESC.get(tag, ""),
                "side": "HOLD",
                "entry_price": None, "SL": None, "TP": None,
                "cond_ok": False, "note": "ENTRY_FIX_PRICE 없음",
                "min_entry_time": ENTRY_FIX_TS
            }

        side_out = plan_side if tag in ("C", "C′") else ("LONG" if tag == "A" else "SHORT")

        # --- 진입가 산정 ---
        if tag in ("A", "B"):
            entry_price = float(ENTRY_FIX_PRICE)  # 28h 이후 첫 봉 시가
            note = "다음봉 시가(고정)"
        else:
            if CUR_28H_CLOSE is None or len(df_best_next) == 0:
                entry_price = float(ENTRY_FIX_PRICE)
                note = "리밋 계산불가→시가(대체)"
            else:
                if tag == "C":
                    target = compute_limit_target_local(
                        "LONG", df_best_next, L, idx_max, idx_min,
                        cur_28h_close_local=CUR_28H_CLOSE,
                        base_hist_close_local=base_hist_close_local  # 0h open 분모
                    )
                else:  # C′
                    target = compute_limit_target_local(
                        "SHORT", df_best_next, L, idx_max, idx_min,
                        cur_28h_close_local=CUR_28H_CLOSE,
                        base_hist_close_local=base_hist_close_local  # 0h open 분모
                    )
                if target is None:
                    entry_price = float(ENTRY_FIX_PRICE)
                    note = "리밋 계산불가→시가(대체)"
                else:
                    entry_price = float(target)
                    note = "되돌림 리밋(고정)"

        # --- SL/TP 계산(ATR) ---
        row_at = df_full[df_full["timestamp"] == ENTRY_FIX_TS] if ENTRY_FIX_TS is not None else pd.DataFrame()
        atr_ref_local = float(row_at["atr"].fillna(method='ffill').iloc[0]) if (not row_at.empty and row_at["atr"].notna().any()) else None
        SL, TP = make_sl_tp(entry_price, side_out, method="ATR", atr=atr_ref_local,
                             sl_pct=None, tp_pct=None, k_sl=k_sl, k_tp=k_tp, tick_size=0.0)

        return {
            "scenario": tag, "설명": STRAT_DESC.get(tag, ""),
            "side": side_out, "entry_price": entry_price, "SL": SL, "TP": TP,
            "cond_ok": cond_ok, "note": note,"min_entry_time": ENTRY_FIX_TS
        }

    # 조건 플래그
    cond_A = (mag >= HI_THR_USE) and up_win and (not regime_down)
    cond_B = (mag >= HI_THR_USE) and dn_win
    cond_C = (LO_THR_USE <= mag < HI_THR_USE) and up_win and (not regime_down)
    cond_Cp = (LO_THR_USE <= mag < HI_THR_USE) and dn_win
    cond_E = (mag < LO_THR_USE) or (best["sim"] < sim_gate)

    if st.button(
        "시나리오 계산",
        help="프리픽스(0~28h)와 매칭 후보를 바탕으로 A~E 시나리오별 고정 진입가/SL/TP·거리(Δ)·퍼센트를 표로 계산합니다."
    ):
        rows = []
        rows.append(scenario_row_now("A", "LONG", cond_A))
        rows.append(scenario_row_now("B", "SHORT", cond_B))
        rows.append(scenario_row_now("C", "LONG", cond_C))
        rows.append(scenario_row_now("C′", "SHORT", cond_Cp))
        rows.append(scenario_row_now("E", "HOLD", cond_E))

        df_scn = pd.DataFrame(rows)

        def _delta(row):
            ep = row.get("entry_price"); sl = row.get("SL"); tp = row.get("TP")
            if ep is None or sl is None or tp is None:
                return pd.Series([None, None, None, None])
            sl_d = abs(ep - sl); tp_d = abs(tp - ep)
            sl_pct_v = (sl_d / ep) * 100.0; tp_pct_v = (tp_d / ep) * 100.0
            return pd.Series([sl_d, tp_d, sl_pct_v, tp_pct_v])

        df_scn[["SL_Δ","TP_Δ","SL_%","TP_%"]] = df_scn.apply(_delta, axis=1)
        show_cols = ["scenario","설명","side","entry_price","SL","TP","SL_Δ","TP_Δ","SL_%","TP_%","cond_ok","min_entry_time","note"]
        df_scn = df_scn[[c for c in show_cols if c in df_scn.columns]]
        st.dataframe(df_scn, use_container_width=True)

# ---------------------------
# LIVE (실거래)
# ---------------------------
elif sim_mode == "LIVE":
    # 정확한 스텝 맞춤을 위한 Decimal 유틸
    from decimal import Decimal, ROUND_DOWN, InvalidOperation

    def _D(x):
        return Decimal(str(x))

    def _fit_step(val, step):
        """step(틱/스텝)에 맞춰 내림 자름 (val/step → 정수화 → *step)"""
        if val is None or not step:
            return val
        v = _D(val)
        s = _D(step)
        return float((v / s).quantize(Decimal("1"), rounding=ROUND_DOWN) * s)

    def _fmt_by_step(val, step):
        """거래소 자릿수에 맞춰 문자열 포맷(가시/디버그용)"""
        if val is None or not step:
            return str(val)
        decs = max(0, -_D(step).as_tuple().exponent)
        return f"{val:.{decs}f}"

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

        # 잔고
        bals = get_futures_balances(tclient)
        def _first_float(d: dict, keys: tuple[str, ...], default: float = 0.0) -> float:
            for k in keys:
                v = d.get(k, None)
                if v not in (None, ""):
                    try:
                        return float(v)
                    except Exception:
                        pass
            return float(default)

        wallet_usdt = _first_float(bals, ("wallet_balance", "walletBalance", "totalWalletBalance", "balance"), 0.0)
        avail_usdt  = _first_float(bals, ("available_balance", "availableBalance", "available", "free", "availableMargin", "crossWalletBalance"), 0.0)

        colb1, colb2 = st.columns(2)
        colb1.metric("USDT Wallet", f"{wallet_usdt:.2f}")
        colb2.metric("USDT Available", f"{avail_usdt:.2f}")

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

    # 기본 필터
    tick_size, qty_step_default = get_symbol_filters(tclient2, symbol=trade_symbol)

    # ★ 중요: exchange_info에서 정확한 필터 재추출(PRICE_FILTER, MARKET_LOT_SIZE, LOT_SIZE, MIN_NOTIONAL, MIN_QTY)
    qty_step_eff, lot_step, market_step = qty_step_default, None, None
    min_notional, min_qty = None, None
    try:
        ex = tclient2.futures_exchange_info()
        sym = next(s for s in ex["symbols"] if s["symbol"] == trade_symbol)
        for f in sym.get("filters", []):
            t = f.get("filterType")
            if t == "PRICE_FILTER":
                tick_size = float(f.get("tickSize", tick_size or 0)) or tick_size
            elif t == "LOT_SIZE":
                lot_step = float(f.get("stepSize", 0)) or None
                min_qty  = float(f.get("minQty", 0) or 0) or min_qty
            elif t == "MARKET_LOT_SIZE":
                market_step = float(f.get("stepSize", 0)) or None
                # 일부 심볼은 여기에도 minQty 있음
                try:
                    mq = f.get("minQty", None)
                    if mq is not None:
                        min_qty = float(mq)
                except Exception:
                    pass
            elif t in ("MIN_NOTIONAL", "NOTIONAL"):  # 구버전/신버전 호환
                try:
                    n = f.get("notional", None) or f.get("minNotional", None)
                    if n is not None:
                        min_notional = float(n)
                except Exception:
                    pass
        # 최종 수량 스텝: MARKET_LOT_SIZE > LOT_SIZE > 기본
        qty_step_eff = market_step or lot_step or qty_step_default
    except Exception:
        qty_step_eff = qty_step_default  # 실패 시 기존값 사용

    # 라운딩 함수(가격/수량)
    def _round_price(p):
        if p is None or not tick_size: return p
        return _fit_step(p, tick_size)

    def _round_qty(q):
        if q is None or not qty_step_eff: return q
        return _fit_step(q, qty_step_eff)

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
        fill_price = entry_price  # HOLD

    if side in ("LONG", "SHORT"):
        sl_raw, tp_raw = make_sl_tp(
            fill_price, side, method=sltp_method, atr=atr_ref,
            sl_pct=None, tp_pct=None, k_sl=k_sl, k_tp=k_tp, tick_size=0.0
        )
        if side == "LONG":
            if sl_raw is not None: sl_raw = min(sl_raw, fill_price)
            if tp_raw is not None: tp_raw = max(tp_raw, fill_price)
        else:
            if sl_raw is not None: sl_raw = max(sl_raw, fill_price)
            if tp_raw is not None: tp_raw = min(tp_raw, fill_price)
    else:
        sl_raw = tp_raw = None

    # 거래소 규격 라운딩(가격)
    entry_rounded = _round_price(fill_price)
    sl_rounded = _round_price(sl_raw) if sl_raw is not None else None
    tp_rounded = _round_price(tp_raw) if tp_raw is not None else None

    # ── 6) 수량 산출(가용잔고 기반) + 라운딩 ────────────────────────
    avail = float(avail_usdt)
    notional = max(0.0, avail) * float(leverage) * (float(size_pct) / 100.0)
    qty_live = (notional / entry_rounded) if (entry_rounded and entry_rounded > 0) else 0.0
    qty_live = _round_qty(qty_live)
    if qty_live is not None and qty_live <= 0 and qty_step_eff:
        qty_live = float(qty_step_eff)

    # ── 7) 사전 검증(표시만) ────────────────────────────────────────
    issues = []
    if side == "LONG":
        if (sl_rounded is None) or (tp_rounded is None) or not (sl_rounded < entry_rounded < tp_rounded):
            issues.append("LONG 조건 위반: SL < Entry < TP 가 보장되지 않았습니다.")
    elif side == "SHORT":
        if (sl_rounded is None) or (tp_rounded is None) or not (tp_rounded < entry_rounded < sl_rounded):
            issues.append("SHORT 조건 위반: TP < Entry < SL 가 보장되지 않았습니다.")
    if qty_live is None or qty_live <= 0:
        issues.append("수량이 0입니다. 잔고/레버리지/사이즈%를 확인하세요.")
    if tick_size is None or qty_step_eff is None:
        issues.append("거래소 필터 조회 실패(tick_size/qty_step). 주문이 거절될 수 있습니다.")

    # ★ 추가 검증: MIN_QTY & MIN_NOTIONAL
    if min_qty is not None and qty_live is not None and qty_live < float(min_qty):
        issues.append(f"수량이 최소 수량(minQty={min_qty}) 미만입니다.")
    if min_notional is not None and entry_rounded is not None and qty_live is not None:
        if (entry_rounded * qty_live) < float(min_notional):
            issues.append(f"명목가치가 최소 기준(minNotional={min_notional}) 미만입니다.")

    # ── 8) 미리보기 ─────────────────────────────────────────────────
    st.markdown("### 📌 주문 미리보기")
    colp1, colp2, colp3, colp4, colp5 = st.columns(5)
    colp1.metric("Side", side)
    colp2.metric("Entry(라운딩)", _fmt_by_step(entry_rounded, tick_size) if entry_rounded else "0")
    colp3.metric("SL(라운딩)", _fmt_by_step(sl_rounded, tick_size) if sl_rounded else "0")
    colp4.metric("TP(라운딩)", _fmt_by_step(tp_rounded, tick_size) if tp_rounded else "0")
    colp5.metric("수량(계약, 라운딩)", _fmt_by_step(qty_live, qty_step_eff) if qty_live else "0")
    st.caption("슬리피지 적용 → 가격/수량 라운딩 → 필터 검증 순으로 계산됩니다.")

    if issues:
        for msg in issues:
            st.warning(f"⚠ {msg}")
    else:
        st.success("검증 통과: 형식/방향 OK (주문 가능 상태)")

    # ── 9) 슬리피지 캡 + TTL ───────────────────────────────────────
    st.divider()
    colo1, colo2 = st.columns(2)
    with colo1:
        slip_cap_pct = st.number_input("슬리피지 캡(%)", 0.0, 5.0, 0.5, 0.1, help="시장가 체결 보호. 예상 진입가 대비 허용 편차 한도.")
    with colo2:
        ttl_min = st.number_input("신호 유효시간(분)", 1, 180, 120, 1, help="발주 버튼 활성화 유효시간(현재 시각 - 엔트리 기준 시각)")
    force_no_slipcap = st.checkbox("슬리피지 캡 무시(긴급 실행)", value=False)

    now_utc = pd.Timestamp.utcnow().tz_localize(None)
    ttl_ok = (now_utc >= t_entry) and ((now_utc - t_entry) <= pd.Timedelta(minutes=float(ttl_min)))

    slip_ok = None
    slip_obs_pct = None
    if now_utc >= t_entry:
        try:
            mp = tclient2.futures_mark_price(symbol=trade_symbol)
            last_px = float(mp["markPrice"])
        except Exception:
            last_px = float(df_full["close"].iloc[-1])
        if entry_rounded:
            slip_obs_pct = abs(entry_rounded - last_px) / entry_rounded * 100.0
            slip_ok = (slip_obs_pct <= float(slip_cap_pct)) or force_no_slipcap
            if (slip_obs_pct > float(slip_cap_pct)) and (not force_no_slipcap):
                issues.append(f"슬리피지 캡 초과: 관측 {slip_obs_pct:.2f}% > 허용 {slip_cap_pct:.2f}%")
    else:
        st.info("엔트리 시각 이전입니다. 엔트리 이후 TTL 내에서만 버튼이 활성화됩니다.")

    # 버튼 상태
    can_order = (side in ("LONG","SHORT")) and (not issues) and ttl_ok

    st.markdown("### 🧾 사전 체크리스트")
    st.write({
        "TTL OK": ttl_ok,
        "Side": side,
        "tick_size/qty_step OK": (tick_size is not None and qty_step_eff is not None),
        "Qty>0": (qty_live is not None and qty_live > 0),
        "SL/TP 방향 보장": (
            (side=="LONG"  and sl_rounded is not None and tp_rounded is not None and sl_rounded<entry_rounded<tp_rounded) or
            (side=="SHORT" and sl_rounded is not None and tp_rounded is not None and tp_rounded<entry_rounded<sl_rounded)
        ),
        "slip_ok": (slip_ok if slip_ok is not None else "N/A"),
        "slip_obs_pct(%)": (round(slip_obs_pct, 3) if slip_obs_pct is not None else "N/A"),
        "qty_step_eff": qty_step_eff,
        "minQty": (min_qty if min_qty is not None else "N/A"),
        "minNotional": (min_notional if min_notional is not None else "N/A"),
    })

    # ── 10) 주문 버튼 & 호출 ───────────────────────────────────────
    order_btn = st.button("🟢 동시 발주 (Entry+SL+TP)", disabled=not can_order, help="검증 통과 시에만 활성화. GTC로 한 번에 제출.")
    if order_btn:
        try:
            # 최종 한 번 더 스텝에 맞춰 자름(precision 에러 방지)
            entry_rounded = _fit_step(entry_rounded, tick_size)
            sl_rounded    = _fit_step(sl_rounded,    tick_size)
            tp_rounded    = _fit_step(tp_rounded,    tick_size)
            qty_live      = _fit_step(qty_live,      qty_step_eff)

            # 최소 조건 재확인
            if (min_qty is not None) and (qty_live is not None) and (qty_live < float(min_qty)):
                raise ValueError(f"수량이 최소 수량(minQty={min_qty}) 미만입니다.")
            if (min_notional is not None) and (entry_rounded is not None) and (qty_live is not None):
                if entry_rounded * qty_live < float(min_notional):
                    raise ValueError(f"명목가치가 최소 기준(minNotional={min_notional}) 미만입니다.")

            # 주문 호출 (place_futures_market_bracket 시그니처에 맞춤)
            resp = place_futures_market_bracket(
                tclient2,
                trade_symbol,
                side,
                float(qty_live),
                float(entry_rounded),                                   # entry_price_ref
                (float(sl_rounded) if sl_rounded is not None else None),# sl_price
                (float(tp_rounded) if tp_rounded is not None else None),# tp_price
                float(qty_step_eff or 0.0),                              # qty_step (시장가면 MARKET_LOT_SIZE)
                float(tick_size or 0.0),                                 # tick_size (PRICE_FILTER)
            )
            st.success("발주 성공: 브래킷 세트가 제출되었습니다.")
            st.json(resp)
        except Exception as e:
            st.error(f"발주 실패: {e}")

    # ── 11) 포지션/미체결 확인 ─────────────────────────────────────
    st.markdown("### 📌 현재 포지션 / 미체결")
    try:
        pos = get_futures_positions(tclient2, symbol=trade_symbol)
        if not pos:
            st.info("열린 포지션이 없습니다.")
        else:
            df_pos = pd.DataFrame(pos)
            for c in ["positionAmt", "entryPrice", "unRealizedProfit", "markPrice", "liquidationPrice"]:
                if c in df_pos.columns:
                    df_pos[c] = pd.to_numeric(df_pos[c], errors="coerce")
            want = ["symbol", "positionAmt", "entryPrice", "unRealizedProfit", "markPrice", "liquidationPrice"]
            show = [c for c in want if c in df_pos.columns]
            st.dataframe(df_pos[show], use_container_width=True)
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
        sim_gate = float(sim_gate_base)
        LO_THR_USE = LO_THR
        HI_THR_USE = HI_THR

        side = "HOLD"
        if best["sim"] >= sim_gate:
            mag_up = abs(max_up); mag_dn = abs(min_dn); mag = max(mag_up, mag_dn)
            if mag >= LO_THR_USE:
                if regime_down and (mag_up >= mag_dn):
                    side = "SHORT"
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
                cur_28h_close = _get_close_at_or_before(df_roll, t_entry)

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

    # ✅ 뉴스/정규화 파이프라인을 NOW-상승과 완전히 동일하게 사용
    #    df_full_static에는 이미 build_news_frame() 결과가 병합되고 z-score/결측 보정이 끝난 상태입니다.
    #    하락/횡보 모드에서도 동일 파이프라인을 재사용합니다.
    POOL_START_B  = pd.Timestamp("2020-01-01 00:00:00")
    POOL_END_B    = pd.Timestamp("2020-11-01 00:00:00")

    # NOW-상승과 동일한 df_full_static(뉴스 병합+정규화 완료본) 재사용
    df_full_b = df_full_static.copy()


    pool_df_b = df_full_b[(df_full_b["timestamp"] >= POOL_START_B) & (df_full_b["timestamp"] < POOL_END_B)].reset_index(drop=True)
    if len(pool_df_b) < window_size:
        st.error("NOW-하락/횡보: 2020 풀 데이터가 부족합니다."); st.stop()

    (ref_start, ref_end), (pred_start, pred_end) = pick_blocks(df_full_b["timestamp"].iloc[-1], step_hours=step_hours)

    # ── 후보 탐색 (DTW) ───────────────────────────────────────────
    cands = get_candidates_a(
        df_pool=pool_df_b,
        ref_range=(ref_start, ref_end),
        df_ref=df_full_b,
        feat_cols=F.FEAT_COLS,
        step_hours=step_hours,
        window_size=window_size,
        sim_mode="DTW", w_dtw=0.5,
        topN=5 if fast else 10,
        ex_margin_days=10 if fast else 5
    )

    results = []
    stepTD = pd.Timedelta(hours=step_hours)
    ENTRY_DELAY_BARS = max(1, int(np.ceil(ENTRY_DELAY_HOURS / 4.0)))  # 28h -> 7 bars(4h)
    for f in cands:
        next_start = f["end"]; next_end = next_start + stepTD
        df_next = pool_df_b[(pool_df_b["timestamp"] >= next_start) & (pool_df_b["timestamp"] < next_end)]
        if len(df_next) < window_size:
            continue

        closes = df_next["close"].to_numpy()
        base_open0 = float(df_next["open"].iloc[0])

        # 과거측 28h 종가(없으면 마지막 종가)
        base_close_28h = float(df_next["close"].iloc[ENTRY_DELAY_BARS - 1]) \
            if len(df_next) >= ENTRY_DELAY_BARS else float(df_next["close"].iloc[-1])

        # 프리픽스 매칭용(원시%, 0h open 분모)
        pct_raw = (closes - base_open0) / base_open0 * 100.0

        results.append({
            "sim": f["sim"],
            "next_start": next_start,
            "next_end": next_end,
            "pct": pct_raw,  # 프리픽스 매칭용
            "df_next": df_next.reset_index(drop=True),
            "base_close": base_open0,        # 0h open
            "base_close_28h": base_close_28h # 28h close
        })

    # ── 현재 진행 프리픽스(0~28h 포함) ─────────────────────────────
    now_ts_b = df_full_b["timestamp"].iloc[-1]
    cur_pred_seg = df_full_b[
        (df_full_b["timestamp"] >= pred_start) &
        (df_full_b["timestamp"] <= min(now_ts_b, pred_end))
    ]
    if len(cur_pred_seg) == 0 or len(results) == 0:
        st.info("데이터 부족"); st.stop()

    # 프리픽스(엔트리 지연 28h까지) 잘라서 코사인 매칭 (open0 기준 유지)
    base_cur_open0 = float(cur_pred_seg["open"].iloc[0])
    prefix_end = min(pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS), pred_end)
    cur_prefix = cur_pred_seg[cur_pred_seg["timestamp"] <= prefix_end]
    a = ((cur_prefix["close"] - base_cur_open0) / base_cur_open0 * 100.0).to_numpy(dtype=float)
    L_use = ENTRY_DELAY_BARS + 1  # 0~7 포함 → 8개
    a = a[:min(L_use, len(a))]
    L = len(a)

    # ── 프리픽스 최고 후보 선택(코사인) ────────────────────────────
    best = None
    for r in results:
        b = np.array(r["pct"], dtype=float)[:L]
        sim_shape = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) else float(cosine_similarity([a], [b])[0][0])
        if (best is None) or (sim_shape > best["sim"]):
            best = {"sim": sim_shape, "flow": r}

    df_best_next = best["flow"]["df_next"]

    # ✅ 표시/그래프용 앵커 정렬: 과거=과거 28h 종가, 현재=현재 28h 종가
    base_hist_close_28h = float(best["flow"].get("base_close_28h", best["flow"]["base_close"]))

    ENTRY_ANCHOR_TS = pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS)
    CUR_28H_CLOSE = _get_close_at_or_before(df_full_b, ENTRY_ANCHOR_TS)

    # 현재 28h 종가가 없으면, 28h 직후 첫 오픈가로 fallback
    _seg_after = df_full_b[df_full_b["timestamp"] > ENTRY_ANCHOR_TS]
    if CUR_28H_CLOSE is None and not _seg_after.empty:
        CUR_28H_CLOSE = float(_seg_after["open"].iloc[0])

    # ✅✅ mag_adj 분모를 NOW-상승과 동일하게 통일:
    # "과거 0h open(= base_close) 우선, 없으면 28h close"
    base_hist_close_local = float(best["flow"].get("base_close", base_hist_close_28h))

    # ── 퍼센트 테이블(표시용): 모두 28h 앵커 기준 ───────────────────
    past_pct_tbl = pd.DataFrame({
        "k": np.arange(len(df_best_next), dtype=int),
        "r_open_%(28h)":  (df_best_next['open']  / base_hist_close_28h - 1.0) * 100.0,
        "r_close_%(28h)": (df_best_next['close'] / base_hist_close_28h - 1.0) * 100.0,
        "r_high_%(28h)":  (df_best_next['high']  / base_hist_close_28h - 1.0) * 100.0,
        "r_low_%(28h)":   (df_best_next['low']   / base_hist_close_28h - 1.0) * 100.0,
    }).reset_index(drop=True)

    now_pct_tbl = pd.DataFrame({
        "k": np.arange(len(cur_pred_seg), dtype=int),
        "r_open_%(28h)":  (cur_pred_seg['open']  / float(CUR_28H_CLOSE) - 1.0) * 100.0 if CUR_28H_CLOSE is not None else np.nan,
        "r_close_%(28h)": (cur_pred_seg['close'] / float(CUR_28H_CLOSE) - 1.0) * 100.0 if CUR_28H_CLOSE is not None else np.nan,
        "r_high_%(28h)":  (cur_pred_seg['high']  / float(CUR_28H_CLOSE) - 1.0) * 100.0 if CUR_28H_CLOSE is not None else np.nan,
        "r_low_%(28h)":   (cur_pred_seg['low']   / float(CUR_28H_CLOSE) - 1.0) * 100.0 if CUR_28H_CLOSE is not None else np.nan,
    }).reset_index(drop=True)

    with st.expander("📊 과거_퍼센트표 (앵커=과거 28h 종가, %)"):
        st.dataframe(past_pct_tbl, use_container_width=True)
    with st.expander("📊 현재_퍼센트표 (앵커=현재 28h 종가, %)"):
        st.dataframe(now_pct_tbl, use_container_width=True)

    st.markdown("### ⏱️ 시간 정보")
    st.write({ "현재 블록 구간": f"{pred_start} ~ {pred_end}" })

    # ── 그래프(28h 기준선) — 곡선도 전부 28h 앵커 기준 ─────────────
    hist_full_28h = ((df_best_next["close"].to_numpy(dtype=float) - base_hist_close_28h) / base_hist_close_28h * 100.0)
    a_plot_28h = ((cur_pred_seg["close"].to_numpy(dtype=float) - float(CUR_28H_CLOSE)) / float(CUR_28H_CLOSE) * 100.0) if CUR_28H_CLOSE is not None else np.array([])

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(np.arange(len(hist_full_28h)), hist_full_28h, label="매칭 72h (28h앵커%)")
    if a_plot_28h.size > 0:
        ax.plot(np.arange(len(a_plot_28h)), a_plot_28h, label=f"현재 진행 (28h앵커%, L={len(a_plot_28h)})")
    ax.axvline(ENTRY_DELAY_BARS, ls="--", label="엔트리 기준(28h)")
    ax.axhline(HI_THR, ls="--"); ax.axhline(-HI_THR, ls="--")
    ax.axhline(LO_THR, ls=":");  ax.axhline(-LO_THR, ls=":")
    ax.axhline(0, ls=":")
    ax.set_title("NOW-하락/횡보: 28h 기준 · 진행 vs 매칭 (모두 28h 앵커%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.caption("세로 점선: 28h(엔트리 기준), 가로 점선 LO/HI: 중·강 임계값")
    st.pyplot(fig)

    # ── 후행 분포/시나리오 판정(프리픽스 상대변화 기반) ────────────
    hist_full = np.array(best["flow"]["pct"], dtype=float)  # 프리픽스 원시% (0h open 분모)
    base_now  = float(hist_full[L - 1]) if L > 0 else 0.0
    fut = hist_full[L - 1:] - base_now
    idx_max = int(np.argmax(fut)) if fut.size > 0 else 0
    idx_min = int(np.argmin(fut)) if fut.size > 0 else 0
    max_up = float(np.max(fut)) if fut.size > 0 else 0.0
    min_dn = float(np.min(fut)) if fut.size > 0 else 0.0

    sim_gate = float(sim_gate_base)
    mag_up = abs(max_up); mag_dn = abs(min_dn); mag = max(mag_up, mag_dn)
    up_win = mag_up >= mag_dn; dn_win = mag_dn > mag_up

    if (mag >= HI_THR) and up_win :
        current_scenario = "A"
    elif (mag >= HI_THR) and dn_win:
        current_scenario = "B"
    elif (LO_THR <= mag < HI_THR) and up_win:
        current_scenario = "C"
    elif (LO_THR <= mag < HI_THR) and dn_win:
        current_scenario = "C′"
    else:
        current_scenario = "E"
    if best["sim"] < sim_gate:
        current_scenario = "E"

    st.markdown(f"### 📌 현재 판정: **{current_scenario} 시나리오**")
    st.caption(f"현재 유사도 = {best['sim']:.3f} / 게이트 = {sim_gate_base:.2f}")
    st.write(f"🕒 현재 데이터 최신 시점: {now_ts_b}")

    STRAT_DESC = {
        "A": "강한 상승: HI_THR 이상 상승 우위 → 다음봉 시가(고정)",
        "B": "강한 하락: HI_THR 이상 하락 우위 → 다음봉 시가(고정)",
        "C": "중간 상승: LO~HI, 상승 우위 → 되돌림 리밋(고정, 터치 미검증/표시용)",
        "C′": "중간 하락: LO~HI, 하락 우위 → 되돌림 리밋(고정, 터치 미검증/표시용)",
        "E": "약함/미달 → HOLD"
    }

    # ── 28h 기준 시각과 고정 엔트리 시각/가격 (표시/기본값) ──────
    ENTRY_FIX_TS = None; ENTRY_FIX_PRICE = None
    if not _seg_after.empty:
        ENTRY_FIX_TS  = _seg_after["timestamp"].iloc[0]        # 28h '직후' 첫 오픈(예: 08:00)
        ENTRY_FIX_PRICE = float(_seg_after["open"].iloc[0])

    # ── 리밋 타깃 산출: 분모=base_hist_close_local(0h open 우선) ────
    def compute_limit_target_local(side: str,
                                   df_next_best: pd.DataFrame,
                                   L_local: int, idx_max_local: int, idx_min_local: int,
                                   cur_28h_close_local: float, base_hist_close_local: float):
        """
        - mag_adj 계산 분모: NOW-상승과 동일하게 과거 0h open(우선), 없으면 28h close
        - 최종 타깃 가격: 현재 28h 앵커(CUR_28H_CLOSE) 기준 적용
        """
        if side == "LONG":
            end_k = min((L_local - 1) + idx_max_local, len(df_next_best) - 1)
            lows_slice = df_next_best["low"].iloc[:end_k + 1].to_numpy(dtype=float)
            if lows_slice.size == 0:
                return None
            low_min = float(np.min(lows_slice))
            drop_pct = (low_min / base_hist_close_local - 1.0) * 100.0
            mag_adj = _adjust_magnitude(abs(drop_pct))
            return cur_28h_close_local * (1.0 - mag_adj / 100.0)
        else:  # SHORT
            end_k = min((L_local - 1) + idx_min_local, len(df_next_best) - 1)
            highs_slice = df_next_best["high"].iloc[:end_k + 1].to_numpy(dtype=float)
            if highs_slice.size == 0:
                return None
            high_max = float(np.max(highs_slice))
            up_pct = (high_max / base_hist_close_local - 1.0) * 100.0
            mag_adj = _adjust_magnitude(abs(up_pct))
            return cur_28h_close_local * (1.0 + mag_adj / 100.0)

    # ── 시나리오 행(고정 엔트리 규칙 / NOW는 터치 미검증) ──────────
    def scenario_row_now(tag: str, plan_side: str, cond_ok: bool):
        if tag == "E":
            return {
                "scenario": tag, "설명": STRAT_DESC.get(tag, ""),
                "side": "HOLD", "entry_price": None, "SL": None, "TP": None,
                "cond_ok": cond_ok, "note": "항상 HOLD", "min_entry_time": ENTRY_FIX_TS
            }

        if ENTRY_FIX_PRICE is None:
            return {
                "scenario": tag, "설명": STRAT_DESC.get(tag, ""),
                "side": "HOLD", "entry_price": None, "SL": None, "TP": None,
                "cond_ok": False, "note": "ENTRY_FIX_PRICE 없음", "min_entry_time": ENTRY_FIX_TS
            }

        side_out = plan_side if tag in ("C", "C′") else ("LONG" if tag == "A" else "SHORT")

        # 진입가 산정
        if tag in ("A", "B"):
            entry_price = float(ENTRY_FIX_PRICE)
            note = "다음봉 시가(고정)"
        else:
            if (CUR_28H_CLOSE is None) or (len(df_best_next) == 0):
                entry_price = float(ENTRY_FIX_PRICE)
                note = "리밋 계산불가→시가(대체)"
            else:
                if tag == "C":
                    target = compute_limit_target_local(
                        "LONG", df_best_next, L, idx_max, idx_min,
                        cur_28h_close_local=float(CUR_28H_CLOSE),
                        base_hist_close_local=base_hist_close_local
                    )
                else:
                    target = compute_limit_target_local(
                        "SHORT", df_best_next, L, idx_max, idx_min,
                        cur_28h_close_local=float(CUR_28H_CLOSE),
                        base_hist_close_local=base_hist_close_local
                    )
                if target is None:
                    entry_price = float(ENTRY_FIX_PRICE)
                    note = "리밋 계산불가→시가(대체)"
                else:
                    entry_price = float(target)
                    note = "되돌림 리밋(고정, 터치 미검증)"

        # SL/TP 계산(ATR) — NOW는 표시용이므로 28h 직후봉 ATR 사용(ENTRY_FIX_TS)
        row_at = df_full_b[df_full_b["timestamp"] == ENTRY_FIX_TS] if ENTRY_FIX_TS is not None else pd.DataFrame()
        atr_ref_local = float(row_at["atr"].fillna(method='ffill').iloc[0]) if (not row_at.empty and row_at["atr"].notna().any()) else None
        SL, TP = make_sl_tp(entry_price, side_out, method="ATR", atr=atr_ref_local,
                             sl_pct=None, tp_pct=None, k_sl=k_sl, k_tp=k_tp, tick_size=0.0)

        return {
            "scenario": tag, "설명": STRAT_DESC.get(tag, ""),
            "side": side_out, "entry_price": entry_price, "SL": SL, "TP": TP,
            "cond_ok": cond_ok, "note": note, "min_entry_time": ENTRY_FIX_TS
        }

    # 조건 플래그
    cond_A  = (mag >= HI_THR) and up_win
    cond_B  = (mag >= HI_THR) and dn_win
    cond_C  = (LO_THR <= mag < HI_THR) and up_win
    cond_Cp = (LO_THR <= mag < HI_THR) and dn_win
    cond_E  = (mag < LO_THR) or (best["sim"] < sim_gate)

    if st.button("시나리오 계산", help="NOW-하락/횡보(고정 엔트리, 앵커=28h 종가)로 A~E 시나리오별 진입가/SL/TP·거리·%를 표로 계산합니다."):
        rows = [
            scenario_row_now("A",  "LONG",  cond_A),
            scenario_row_now("B",  "SHORT", cond_B),
            scenario_row_now("C",  "LONG",  cond_C),
            scenario_row_now("C′", "SHORT", cond_Cp),
            scenario_row_now("E",  "HOLD",  cond_E),
        ]
        df_scn = pd.DataFrame(rows)

        def _delta(row):
            ep, sl, tp = row.get("entry_price"), row.get("SL"), row.get("TP")
            if ep is None or sl is None or tp is None:
                return pd.Series([None, None, None, None])
            sl_d = abs(ep - sl); tp_d = abs(tp - ep)
            sl_pct_v = (sl_d / ep) * 100.0; tp_pct_v = (tp_d / ep) * 100.0
            return pd.Series([sl_d, tp_d, sl_pct_v, tp_pct_v])

        df_scn[["SL_Δ","TP_Δ","SL_%","TP_%"]] = df_scn.apply(_delta, axis=1)
        show_cols = ["scenario","설명","side","entry_price","SL","TP","SL_Δ","TP_Δ","SL_%","TP_%","cond_ok","min_entry_time","note"]
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

    # --- 고정 파라미터 (fast) ---
    sim_gate_base = 0.75
    topN = 5
    ex_margin_days = 10
    ROLL_START_B = pd.Timestamp("2025-01-01 00:00:00")
    step_hours = 72
    window_size = 18
    ENTRY_DELAY_HOURS = 28
    stepTD  = pd.Timedelta(hours=step_hours)
    delayTD = pd.Timedelta(hours=ENTRY_DELAY_HOURS)

    # --- 풀 범위 (2020) ---
    SCALE_END_B  = pd.Timestamp("2020-11-01 00:00:00")
    POOL_START_B = pd.Timestamp("2020-01-01 00:00:00")
    POOL_END_B   = pd.Timestamp("2020-11-01 00:00:00")

    # --- 전처리 ---
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

        # --- 후보 탐색 ---
        cands = get_candidates_a(
            df_pool=pool_df_b,
            ref_range=(ref_b["start"], ref_b["end"]),
            df_ref=df_full_b,
            feat_cols=F.FEAT_COLS,
            step_hours=step_hours, window_size=window_size,
            sim_mode=sim_engine, w_dtw=0.5,  # Hybrid 없음 → w_dtw 무시
            topN=topN, ex_margin_days=ex_margin_days
        )
        if not cands:
            pbar.progress(int(100 * (j + 1) / max(1, total))); continue

        # ---------- 결과 후보 구성 (28h 종가 포함) ----------
        ENTRY_DELAY_BARS = max(1, int(np.ceil(ENTRY_DELAY_HOURS / 4.0)))  # 28h -> 7 bars(4h봉)
        results = []
        for f in cands:
            next_start = f["end"]; next_end = next_start + stepTD
            df_next = pool_df_b[(pool_df_b["timestamp"] >= next_start) & (pool_df_b["timestamp"] < next_end)]
            if len(df_next) < window_size:
                continue
            closes = df_next["close"].to_numpy()
            baseC_open0  = float(df_next["open"].iloc[0])  # 0h open
            baseC_28h = float(df_next["close"].iloc[ENTRY_DELAY_BARS - 1]) \
                if len(df_next) >= ENTRY_DELAY_BARS else float(df_next["close"].iloc[-1])
            pct_c  = (closes - baseC_open0) / baseC_open0 * 100.0  # 프리픽스 매칭용(원시%)

            results.append({
                "sim": f["sim"],
                "next_start": next_start, "next_end": next_end,
                "pct": pct_c,
                "df_next": df_next.reset_index(drop=True),
                "base_close": baseC_open0,     # 0h open
                "base_close_28h": baseC_28h,   # 28h close
            })
        if not results:
            pbar.progress(int(100 * (j + 1) / max(1, total))); continue

        # --- 현재 프리픽스 (0~28h 포함) ---
        pred_seg = df_roll_base[(df_roll_base["timestamp"] >= pred_b["start"]) & (df_roll_base["timestamp"] <= t_entry)]
        if len(pred_seg) == 0:
            pbar.progress(int(100 * (j + 1) / max(1, total))); continue

        base_cur = float(pred_seg["close"].iloc[0])
        a = ((pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)
        L = len(a)

        # --- 프리픽스 최고 후보 (코사인/DTW-코사인) ---
        best = None
        for r in results:
            b = np.array(r["pct"], dtype=float)[:L]
            sim_shape = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) else float(cosine_similarity([a],[b])[0][0])
            if (best is None) or (sim_shape > best["sim"]):
                best = {"sim": sim_shape, "flow": r}

        # --- 후행 분포 / 시나리오 방향성 ---
        hist_full = np.array(best["flow"]["pct"], dtype=float)
        base_now  = float(hist_full[L - 1]) if L > 0 else 0.0
        fut       = hist_full[L - 1:] - base_now
        idx_max   = int(np.argmax(fut)) if fut.size > 0 else 0
        idx_min   = int(np.argmin(fut)) if fut.size > 0 else 0
        max_up    = float(np.max(fut))  if fut.size > 0 else 0.0
        min_dn    = float(np.min(fut))  if fut.size > 0 else 0.0

        sim_gate = float(sim_gate_base)
        side = "HOLD"
        if best["sim"] >= sim_gate:
            mag_up = abs(max_up); mag_dn = abs(min_dn); mag = max(mag_up, mag_dn)
            if mag >= LO_THR:
                side = "LONG" if mag_up >= mag_dn else "SHORT"

        # --- 엔트리 산정: HI 구간은 다음봉 시가, 그 외는 되돌림 리밋 ---
        entry_time = entry_price = entry_target = None
        if side in ("LONG","SHORT"):
            if max(abs(max_up), abs(min_dn)) >= HI_THR:
                etime, eprice = make_entry_at(df_roll_base, t_entry, rule="다음봉 시가")
                if etime is not None and etime < t_entry:
                    seg_after = df_roll_base[df_roll_base["timestamp"] > t_entry]
                    if not seg_after.empty:
                        etime = seg_after["timestamp"].iloc[0]
                        eprice = float(seg_after["open"].iloc[0])
                entry_time, entry_price = etime, eprice
            else:
                # ====== 리밋 타깃: NOW-하락과 동일한 분모 규칙 적용 ======
                df_next_best = best["flow"]["df_next"]

                # (표시/참조용) 28h 종가
                base_hist_close_28h = float(best["flow"].get("base_close_28h", best["flow"]["base_close"]))
                # ✅ mag_adj 분모 통일: "과거 0h open(=base_close) 우선, 없으면 28h close"
                base_hist_close_local = float(best["flow"].get("base_close", base_hist_close_28h))

                cur_28h_close = _get_close_at_or_before(df_roll_base, t_entry)
                if (cur_28h_close is not None) and (len(df_next_best) > 0):
                    if side == "LONG":
                        end_k = min((L - 1) + idx_max, len(df_next_best) - 1)
                        lows_slice = df_next_best["low"].iloc[:end_k + 1].to_numpy(dtype=float)
                        if lows_slice.size > 0:
                            low_min = float(np.min(lows_slice))
                            # 🔁 분모 통일
                            drop_pct = (low_min / base_hist_close_local - 1.0) * 100.0
                            mag_adj = _adjust_magnitude(abs(drop_pct))
                            entry_target = cur_28h_close * (1.0 - mag_adj/100.0)
                            entry_time, entry_price = _touch_entry(
                                df_roll_base, t_entry, pred_b["end"], "LONG", entry_target
                            )
                    else:  # SHORT
                        end_k = min((L - 1) + idx_min, len(df_next_best) - 1)
                        highs_slice = df_next_best["high"].iloc[:end_k + 1].to_numpy(dtype=float)
                        if highs_slice.size > 0:
                            high_max = float(np.max(highs_slice))
                            # 🔁 분모 통일
                            up_pct = (high_max / base_hist_close_local - 1.0) * 100.0
                            mag_adj = _adjust_magnitude(abs(up_pct))
                            entry_target = cur_28h_close * (1.0 + mag_adj/100.0)
                            entry_time, entry_price = _touch_entry(
                                df_roll_base, t_entry, pred_b["end"], "SHORT", entry_target
                            )
                # ====== /리밋 타깃 ======

        # --- ATR 참조 ---
        atr_ref = None
        if entry_time is not None:
            row_at = df_roll_base[df_roll_base["timestamp"] == entry_time]
            if not row_at.empty and row_at["atr"].notna().any():
                atr_ref = float(row_at["atr"].fillna(method='ffill').iloc[0])

        # --- SL/TP 계산 및 시뮬레이션 ---
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
            size = float(eq_run) * float(max_leverage)   # 단순 레버리지 캡
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

        # --- 에쿼티 업데이트 ---
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

    # --- 결과 표시 ---
    df_log = pd.DataFrame(trade_logs).sort_values("pred_start").reset_index(drop=True)

    df_show = (df_log.copy()
               .drop(columns=["gross_ret_%","net_ret_%"], errors="ignore")
               .rename(columns={"ret_equity_%": "ret_%(levered)"}))
    cols = ["pred_start","pred_end","t_entry","side","entry_time","entry","entry_target",
            "SL","TP","size_notional","used_lev","cap_hit","pnl_usd","ret_%(levered)",
            "eq_before","eq_after","exit_time","exit","sim_prefix"]
    df_show = df_show[[c for c in cols if c in df_show.columns]]

    st.markdown("### 결과 테이블 (레버리지 반영 수익률) — B")
    st.caption(
        "ret_%(levered) = net_ret_% × (size_notional / eq_before) · "
        "리밋 분모: 과거 0h 오픈(우선, base_close) → 없으면 28h 종가(base_close_28h)"
    )
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
