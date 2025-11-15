# app.py (전체)  -- 수정본 (튜너용 evaluate_wrapper + run_backtest_with_params 포함)
# 요청 반영:
# 1) LIVE 모드 제거, 대신 "6M-상승" 모드 추가(최근 6개월 백테스트 및 BT-상승과 비교)
# 2) 튜닝 종료 시 자동 저장(세션 및 tuned_params.json), "즉시 사용/파일 저장/불러오기" 버튼 제거
# 3) BT-상승의 백테스트 시작점(ROLL_START)은 기존 고정값 유지 (사용자 요청: 바꾸지 않음)
# 4) 튜닝 학습/평가는 "최근 6개월"로 고정

# === Prelude: 한글 폰트/음수 디폴트 적용 (UI 없음) ===
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False      # 음수 기호 깨짐 방지
import random
import inspect
import streamlit as st
import numpy as np
import pandas as pd
import json
import time
from collections import deque
from pathlib import Path
from datetime import timedelta
from datetime import date
# === 프로젝트 모듈 ===
from connectors import (
    connect_binance, connect_binance_trade,
    get_futures_balances, get_futures_positions,
    ensure_leverage_and_margin, get_symbol_filters,
)
from data_fetch import fetch_futures_4h_klines, fetch_funding_rate
from features import (
    add_features, apply_static_zscore, finalize_preprocessed,
    window_is_finite, window_vector, GLOBAL_Z_COLS, FEAT_COLS,
)
from similarity import sim_tier3
from blocks import pick_blocks, enumerate_blocks
from trading_utils import (
    make_entry_at, make_sl_tp, position_size, simulate_trade, place_futures_market_bracket,
)
from backtest_utils import build_equity_curve, calc_metrics
from sklearn.metrics.pairwise import cosine_similarity

# === 베이즈 튜너(메타 모델) ===
from tuner import run_bayes_opt

# ---------------------------
# 기본 UI 설정
# ---------------------------
st.set_page_config(page_title="BTC 패턴매칭 전략 스튜디오", page_icon="📊", layout="wide")
st.title("📈 BTC 패턴매칭 전략 스튜디오")

# ---------------------------
# 공통 하이퍼파라미터 (기본값)
# ---------------------------
step_hours = 72
window_size = 18
ENTRY_DELAY_HOURS = 28
ENTRY_RULE_FIXED = "다음봉 시가"
LO_THR = 1.0
HI_THR = 3.0
sim_gate_base = 0.75

STRAT_SLTPS = {
    "A": {"method": "ATR", "k_sl": 1.0, "k_tp": 2.5, "sl_pct": None, "tp_pct": None},
    "B": {"method": "ATR", "k_sl": 1.0, "k_tp": 2.5, "sl_pct": None, "tp_pct": None},
    "C": {"method": "ATR", "k_sl": 1.5, "k_tp": 1.5, "sl_pct": None, "tp_pct": None},
    "C′": {"method": "ATR", "k_sl": 1.5, "k_tp": 1.5, "sl_pct": None, "tp_pct": None},
    "E": {"method": "ATR", "k_sl": None, "k_tp": None, "sl_pct": None, "tp_pct": None},  # HOLD
}

# ---------------------------
# 상단: 모드 선택 + 공통 파라미터
# ---------------------------
colA, colB, colC = st.columns(3)
with colA:
    sim_mode = st.radio(
        "모드",
        ["NOW-상승", "BT-상승", "6M-상승","오늘의 운세"],  # LIVE 제거, 6M-상승 추가
        index=0,
        horizontal=True,
        help="NOW-상승: 단일·32h / BT-상승: 고정 시계열 백테스트 / 6M-상승: 최근 6개월 백테스트"
    )

# 공통 디폴트 (필요시 각 모드에서 재설정)
sim_engine = "DTW"  # ROLLING 계열에서만 사용
w_dtw = 0.5         # Hybrid 제거되었지만 호출 시 인자형 유지(무시됨)

# SL/TP은 기본 ATR 사용.
sltp_method = "ATR"
k_sl = 1.5
k_tp = 2.5

fee_entry = 0.0004
fee_exit = 0.0005
slip_entry = 0.0003
slip_exit = 0.0005

equity = 1000.0
max_leverage = 10.0
def _load_tuned_params_into_session():
    try:
        base = Path(__file__).parent
    except NameError:
        base = Path(".")  # Streamlit Cloud 등 __file__ 없는 환경 대비
    p = base / "tuned_params.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                params = json.load(f)
            # 세션에 주입 (이미 있으면 덮어쓰지 않음)
            st.session_state.setdefault("tuned_params", params)
            st.session_state.setdefault("last_best_params", params)
            st.session_state["tuned_params_path"] = str(p)
            return True
        except Exception as e:
            st.warning(f"tuned_params.json 로드 실패: {e}")
    return False

_loaded_tp = _load_tuned_params_into_session()
if _loaded_tp:
    st.caption(f"🧠 tuned_params.json 로드됨 → {st.session_state['tuned_params_path']}")
# ---------------------------
# 🔌 튜닝값 사용 토글 + 전역 주입
# ---------------------------
use_tuned = st.toggle(
    "🧠 튜닝값 사용(BO/Surrogate)",
    value=bool(st.session_state.get("tuned_params")),  # ← 로드되면 기본 ON
    help="튜닝 섹션에서 저장된 best params를 NOW/BT-상승/6M-상승에 주입"
)
tuned = st.session_state.get("tuned_params")
if use_tuned and tuned:
    # 게이트/지연
    sim_gate_base = float(tuned.get("sim_gate", sim_gate_base))
    ENTRY_DELAY_HOURS = float(tuned.get("delay_h", ENTRY_DELAY_HOURS))

    # 태그별 k 설정
    STRAT_SLTPS["A"]["k_sl"] = STRAT_SLTPS["B"]["k_sl"] = float(tuned.get("k_sl_A", STRAT_SLTPS["A"]["k_sl"]))
    STRAT_SLTPS["A"]["k_tp"] = STRAT_SLTPS["B"]["k_tp"] = float(tuned.get("k_tp_A", STRAT_SLTPS["A"]["k_tp"]))
    STRAT_SLTPS["C"]["k_sl"] = STRAT_SLTPS["C′"]["k_sl"] = float(tuned.get("k_sl_C", STRAT_SLTPS["C"]["k_sl"]))
    STRAT_SLTPS["C"]["k_tp"] = STRAT_SLTPS["C′"]["k_tp"] = float(tuned.get("k_tp_C", STRAT_SLTPS["C"]["k_tp"]))

    st.caption(
        f"튜닝 적용 → sim_gate={sim_gate_base:.3f}, delay_h={ENTRY_DELAY_HOURS:.0f}, "
        f"A/B k_sl={STRAT_SLTPS['A']['k_sl']:.2f}, k_tp={STRAT_SLTPS['A']['k_tp']:.2f}, "
        f"C/C′ k_sl={STRAT_SLTPS['C']['k_sl']:.2f}, k_tp={STRAT_SLTPS['C']['k_tp']:.2f}"
    )

# ---------------------------
# BT-상승/6M-상승 모드에서만 노출되는 세부 입력
# ---------------------------
if sim_mode in ("BT-상승", "6M-상승"):
    with colA:
        sim_engine = st.selectbox(
            "유사도 방식", ["DTW", "Cosine"], index=0,
            help="과거 구간과의 유사도 계산 메트릭. DTW(동적 타임워핑) 또는 Cosine(코사인 유사도)만 허용."
        )

        if use_tuned and tuned:
            st.text_input("A/B SL(×ATR)", value=f"{STRAT_SLTPS['A']['k_sl']:.2f}", disabled=True)
            st.text_input("A/B TP(×ATR)", value=f"{STRAT_SLTPS['A']['k_tp']:.2f}", disabled=True)
            st.text_input("C/C′ SL(×ATR)", value=f"{STRAT_SLTPS['C']['k_sl']:.2f}", disabled=True)
            st.text_input("C/C′ TP(×ATR)", value=f"{STRAT_SLTPS['C']['k_tp']:.2f}", disabled=True)
            A_sl = STRAT_SLTPS['A']['k_sl']; A_tp = STRAT_SLTPS['A']['k_tp']
            C_sl = STRAT_SLTPS['C']['k_sl']; C_tp = STRAT_SLTPS['C']['k_tp']
        else:
            A_sl = st.number_input("A/B SL(×ATR)", 0.1, 50.0, 1.0, 0.1)
            A_tp = st.number_input("A/B TP(×ATR)", 0.1, 50.0, 2.5, 0.1)
            C_sl = st.number_input("C/C′ SL(×ATR)", 0.1, 50.0, 1.5, 0.1)
            C_tp = st.number_input("C/C′ TP(×ATR)", 0.1, 50.0, 1.5, 0.1)

    with colB:
        fee_entry = st.number_input("진입 수수료(%)", 0.0, 1.0, 0.04, 0.01, help="백테스트 체결 현실화용 가정 수수료.") / 100.0
        fee_exit  = st.number_input("청산 수수료(%)", 0.0, 1.0, 0.05, 0.01, help="백테스트 체결 현실화용 가정 수수료.") / 100.0
        slip_entry = st.number_input("진입 슬리피지(%)", 0.0, 0.5, 0.03, 0.01, help="체결 미끄러짐 가정치(%).") / 100.0
        slip_exit  = st.number_input("청산 슬리피지(%)", 0.0, 0.5, 0.05, 0.01, help="체결 미끄러짐 가정치(%).") / 100.0

    with colC:
        equity = st.number_input("가상 Equity (USDT)", 10.0, value=1000.0, step=10.0, help="백테스트/포지션 사이징용 가상 잔고")
        max_leverage = st.number_input("최대 레버리지(x)", 1.0, 50.0, 10.0, 1.0, help="사이징 계산 시 사용할 레버리지 상한")
    # =========================
    if "quiz_seed" not in st.session_state:
        st.session_state.quiz_seed = random.randint(0, 2)
    if "quiz_answer" not in st.session_state:
        st.session_state.quiz_answer = None
    if "quiz_shown" not in st.session_state:
        st.session_state.quiz_shown = False

    # =========================
    # 퀴즈 목록
    # =========================
    quizzes = [
        {"question": "비트코인의 창시자는 누구일까요?",
        "options": ["비탈릭 부테린", "일론 머스크", "사토시 나카모토"],
        "answer": "사토시 나카모토"},
        {"question": "비트코인의 최대 발행량은?",
        "options": ["21만 개", "2,100만 개", "2,100억 개"],
        "answer": "2,100만 개"},
        {"question": "비트코인 첫 블록의 이름은?",
        "options": ["제네시스 블록", "오리지널 블록", "알파 블록"],
        "answer": "제네시스 블록"}
    ]
    quiz = quizzes[st.session_state.quiz_seed]

    # =========================
    # 퀴즈 자동 표시 (버튼 없이)
    # =========================
    st.subheader("퀴즈")
    st.text(quiz["question"])
    for i, opt in enumerate(quiz["options"], start=1):
        st.text(f"{i}. {opt}")
    st.info("정답은 백테스트가 끝난 후 공개됩니다.")
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
    st.error("데이터가 부족합니다.")
    st.stop()

# ---------------------------
# 공통 유틸 함수들
# ---------------------------
def get_candidates(df, ref_range, ex_margin_days=5, topN=10, past_only=False):
    ref_seg = df[(df["timestamp"] >= ref_range[0]) & (df["timestamp"] < ref_range[1])]
    if len(ref_seg) < window_size:
        return []
    wL = ref_seg.iloc[:window_size]
    if not window_is_finite(wL):
        return []
    vec_ref = window_vector(wL, L=window_size)
    blocks = enumerate_blocks(df, step_hours=step_hours, window_size=window_size)
    ex_margin = pd.Timedelta(days=ex_margin_days)
    F = len(FEAT_COLS); cand = []
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

def _adjust_magnitude(pct_mag: float) -> float:
    return max(0.0, pct_mag-0.1)

def _get_close_at_or_before(df: pd.DataFrame, ts: pd.Timestamp):
    """ts가 봉 오픈타임이라면 '직전 봉 종가'를 반환. 없으면 가능한 합리적 fallback."""
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
    if seg.empty:
        return None, None
    if side == "LONG":
        hit = seg[seg["low"] <= target_price]
        return (hit["timestamp"].iloc[0], float(target_price)) if not hit.empty else (None, None)
    else:
        hit = seg[seg["high"] >= target_price]
        return (hit["timestamp"].iloc[0], float(target_price)) if not hit.empty else (None, None)

def _resolve_sltp_by_tag(tag: str, default_method: str, default_k_sl: float, default_k_tp: float, strat_sltps_override: dict = None):
    """전략 태그(A/B/C/C′/E)에 맞는 SL/TP 파라미터를 리턴."""
    if strat_sltps_override is None:
        cfg = STRAT_SLTPS.get(tag, {})
    else:
        cfg = strat_sltps_override.get(tag, STRAT_SLTPS.get(tag, {}))
    method = cfg.get("method", default_method)
    if method.upper() == "PCT":
        return {
            "method": "PCT", "k_sl": None, "k_tp": None,
            "sl_pct": cfg.get("sl_pct", None), "tp_pct": cfg.get("tp_pct", None),
        }
    else:
        return {
            "method": "ATR",
            "k_sl": cfg.get("k_sl", default_k_sl), "k_tp": cfg.get("k_tp", default_k_tp),
            "sl_pct": None, "tp_pct": None,
        }

# =========================
# NOW-상승
# =========================
if sim_mode == "NOW-상승":
    st.subheader("NOW-상승: 32h 지연 엔트리 · 1회 거래 (태그별 전략 명시 포함)")
    df_full = df_full_static  # NOW는 static 기준 사용

    # 후보 탐색
    cands = get_candidates(
        df_full, (ref_start, ref_end), ex_margin_days=10, topN=5, past_only=True
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
        base = float(df_next["close"].iloc[0])
        pct_raw = (closes - base) / base * 100.0
        ENTRY_DELAY_BARS = max(1, int(np.ceil(ENTRY_DELAY_HOURS / 4)))
        base_close_28h = float(df_next["close"].iloc[ENTRY_DELAY_BARS - 1]) \
            if len(df_next) >= ENTRY_DELAY_BARS else float(df_next["close"].iloc[-1])
        results.append({
            "sim": f["sim"], "next_start": next_start, "next_end": next_end,
            "pct": pct_raw, "df_next": df_next.reset_index(drop=True),
            "base_close": base, "base_close_28h": base_close_28h
        })

    # 현재 진행 퍼센트 시계열
    cur_pred_seg = df_full[
        (df_full["timestamp"] >= pred_start) & (df_full["timestamp"] <= min(now_ts, pred_end))
    ]
    if len(cur_pred_seg) == 0 or len(results) == 0:
        st.info("데이터 부족")
        st.stop()

    base_cur = float(cur_pred_seg["open"].iloc[0])
    a_plot = ((cur_pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)

    ENTRY_DELAY_BARS = int(np.ceil(ENTRY_DELAY_HOURS / 4.0))
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
    base_hist_close = best["flow"].get(
        "base_close",
        best["flow"].get(
            "base_close_28h",
            (float(df_best_next["close"].iloc[L-1]) if len(df_best_next) >= L and L > 0 else float(df_best_next["close"].iloc[-1]))
        )
    )

    # 표 (퍼센트 테이블)
    past_pct_tbl = pd.DataFrame({
        "k": np.arange(len(df_best_next), dtype=int),
        "r_open_%": (df_best_next['open'] / df_best_next['open'].iloc[0] - 1.0) * 100.0,
        "r_close_%": (df_best_next['close'] / df_best_next['open'].iloc[0] - 1.0) * 100.0,
        "r_high_%": (df_best_next['high'] / df_best_next['open'].iloc[0] - 1.0) * 100.0,
        "r_low_%": (df_best_next['low'] / df_best_next['open'].iloc[0] - 1.0) * 100.0,
    }).reset_index(drop=True)

    now_pct_tbl = pd.DataFrame({
        "k": np.arange(len(cur_pred_seg), dtype=int),
        "r_open_%": (cur_pred_seg['open'] / cur_pred_seg['open'].iloc[0] - 1.0) * 100.0,
        "r_close_%": (cur_pred_seg['close'] / cur_pred_seg['open'].iloc[0] - 1.0) * 100.0,
        "r_high_%": (cur_pred_seg['high'] / cur_pred_seg['open'].iloc[0] - 1.0) * 100.0,
        "r_low_%": (cur_pred_seg['low'] / cur_pred_seg['open'].iloc[0] - 1.0) * 100.0,
    }).reset_index(drop=True)

    with st.expander("📊 과거_퍼센트표 (앵커=과거 0~4h 시가, 원시%)", expanded=False):
        st.dataframe(past_pct_tbl, use_container_width=True)
    with st.expander("📊 현재_퍼센트표 (앵커=현재 0~4h 시가, 원시%)", expanded=False):
        st.dataframe(now_pct_tbl, use_container_width=True)

    st.markdown("### ⏱️ 시간 정보")
    st.write({ "현재 블록 구간": f"{pred_start} ~ {pred_end}" })

    # 그래프
    fig, ax = plt.subplots(figsize=(9, 3))
    hist_full = np.array(best["flow"]["pct"], dtype=float)
    ax.plot(np.arange(len(hist_full)), hist_full, label="매칭 72h(원시%)")
    ax.plot(np.arange(len(a_plot)), a_plot, label=f"현재 진행 (L={len(a_plot)})")
    ax.axvline(L - 1, ls="--", label="엔트리 기준(32h)")
    ax.axhline(HI_THR, ls="--"); ax.axhline(-HI_THR, ls="--")
    ax.axhline(LO_THR, ls=":"); ax.axhline(-LO_THR, ls=":")
    ax.axhline(0, ls=":")
    ax.set_title("NOW-상승: 32h 기준 · 진행 vs 매칭 (원시%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.caption("세로 점선: 32h(엔트리 기준), 가로 점선 LO/HI: 중·강 임계값, 곡선: 프리픽스/후보 원시%")
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
        (df_full["timestamp"] >= ext_start) & (df_full["timestamp"] <= prefix_end)
    ].reset_index(drop=True)
    used_ext = (len(ext_seg) >= 2)
    seg = ext_seg if len(ext_seg) >= 2 else cur_prefix
    anchor = float(seg["close"].iloc[0])
    last = float(seg["close"].iloc[-1])
    ret_pct = (last / anchor - 1.0) * 100.0

    thr_ext = -1.0
    thr_cur = 0.0
    cutoff = (thr_ext if used_ext else thr_cur)
    regime_down = (ret_pct < cutoff)

    sim_gate = float(sim_gate_base)
    LO_THR_USE = LO_THR
    HI_THR_USE = HI_THR

    mag_up = abs(max_up); mag_dn = abs(min_dn); mag = max(mag_up, mag_dn)
    up_win = mag_up >= mag_dn
    dn_win = mag_dn > mag_up

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

    def compute_limit_target_local(side: str, df_next_best: pd.DataFrame, L_local: int, idx_max_local: int, idx_min_local: int, cur_28h_close_local: float, base_hist_close_local: float):
        if side == "LONG":
            end_k = min((L_local - 1) + idx_max_local, len(df_next_best) - 1)
            lows_slice = df_next_best["low"].iloc[:end_k + 1].to_numpy(dtype=float)
            if lows_slice.size == 0:
                return None
            low_min = float(np.min(lows_slice))
            drop_pct = (low_min / base_hist_close_local - 1.0) * 100.0
            mag_adj = _adjust_magnitude(abs(drop_pct))
            return cur_28h_close_local * (1.0 + (-mag_adj) / 100.0)
        elif side == "SHORT":
            end_k = min((L_local - 1) + idx_min_local, len(df_next_best) - 1)
            highs_slice = df_next_best["high"].iloc[:end_k + 1].to_numpy(dtype=float)
            if highs_slice.size == 0:
                return None
            high_max = float(np.max(highs_slice))
            up_pct = (high_max / base_hist_close_local - 1.0) * 100.0
            mag_adj = _adjust_magnitude(abs(up_pct))
            return cur_28h_close_local * (1.0 + mag_adj / 100.0)
        return None

    ENTRY_DELAY_BARS = int(np.ceil(ENTRY_DELAY_HOURS / 4.0))
    ENTRY_ANCHOR_TS = pred_start + pd.Timedelta(hours=ENTRY_DELAY_HOURS)
    _seg_after = df_full[df_full["timestamp"] > ENTRY_ANCHOR_TS]
    if _seg_after.empty:
        ENTRY_FIX_TS, ENTRY_FIX_PRICE = (None, None)
    else:
        ENTRY_FIX_TS = _seg_after["timestamp"].iloc[0]
        ENTRY_FIX_PRICE = float(_seg_after["open"].iloc[0])
    CUR_28H_CLOSE = _get_close_at_or_before(df_full, ENTRY_ANCHOR_TS)
    if CUR_28H_CLOSE is None and ENTRY_FIX_PRICE is not None:
        CUR_28H_CLOSE = float(ENTRY_FIX_PRICE)
    base_hist_close_local = float(base_hist_close)

    def scenario_row_now(tag: str, plan_side: str, cond_ok: bool):
        """ - A/B: 28h 이후 '첫 봉 시가(ENTRY_FIX_PRICE)'로 고정 진입.
            - C/C′: 되돌림 리밋 타깃을 '항상' 진입가로 고정(터치 여부 무관).
        """
        if tag == "E":
            return {
                "scenario": tag, "설명": STRAT_DESC.get(tag, ""), "side": "HOLD",
                "min_entry_time": ENTRY_FIX_TS, "entry_price": None, "SL": None, "TP": None,
                "cond_ok": cond_ok, "note": "항상 HOLD"
            }
        if ENTRY_FIX_PRICE is None:
            return {
                "scenario": tag, "설명": STRAT_DESC.get(tag, ""), "side": "HOLD",
                "entry_price": None, "SL": None, "TP": None, "cond_ok": False, "note": "ENTRY_FIX_PRICE 없음",
                "min_entry_time": ENTRY_FIX_TS
            }

        side_out = plan_side if tag in ("C", "C′") else ("LONG" if tag == "A" else "SHORT")

        # 진입가 산정
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
                        cur_28h_close_local=CUR_28H_CLOSE, base_hist_close_local=base_hist_close_local
                    )
                else:  # C′
                    target = compute_limit_target_local(
                        "SHORT", df_best_next, L, idx_max, idx_min,
                        cur_28h_close_local=CUR_28H_CLOSE, base_hist_close_local=base_hist_close_local
                    )
                if target is None:
                    entry_price = float(ENTRY_FIX_PRICE)
                    note = "리밋 계산불가→시가(대체)"
                else:
                    entry_price = float(target)
                    note = "되돌림 리밋(고정)"

        # SL/TP 계산(ATR)
        row_at = df_full[df_full["timestamp"] == ENTRY_FIX_TS] if ENTRY_FIX_TS is not None else pd.DataFrame()
        atr_ref_local = float(row_at["atr"].fillna(method='ffill').iloc[0]) if (not row_at.empty and row_at["atr"].notna().any()) else None
        param = _resolve_sltp_by_tag(tag, default_method=sltp_method, default_k_sl=k_sl, default_k_tp=k_tp)
        SL, TP = make_sl_tp(
            entry_price, side_out, method=param["method"], atr=atr_ref_local,
            sl_pct=param["sl_pct"], tp_pct=param["tp_pct"],
            k_sl=param["k_sl"], k_tp=param["k_tp"], tick_size=0.0
        )
        return {
            "scenario": tag, "설명": STRAT_DESC.get(tag, ""), "side": side_out,
            "entry_price": entry_price, "SL": SL, "TP": TP, "cond_ok": cond_ok,
            "note": note, "min_entry_time": ENTRY_FIX_TS
        }

    # 조건 플래그
    cond_A  = (mag >= HI_THR) and up_win and (not regime_down)
    cond_B  = (mag >= HI_THR) and dn_win
    cond_C  = (LO_THR <= mag < HI_THR) and up_win and (not regime_down)
    cond_Cp = (LO_THR <= mag < HI_THR) and dn_win
    cond_E  = (mag < LO_THR) or (best["sim"] < sim_gate)

    if st.button("시나리오 계산", help="프리픽스(0~28h)와 매칭 후보를 바탕으로 A~E 시나리오 표 계산"):
        rows = [
            scenario_row_now("A", "LONG", cond_A),
            scenario_row_now("B", "SHORT", cond_B),
            scenario_row_now("C", "LONG", cond_C),
            scenario_row_now("C′", "SHORT", cond_Cp),
            scenario_row_now("E", "HOLD", cond_E),
        ]
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

# =========================
# 공통 백테스트 함수
# =========================
def run_backtest_with_params(
    df_full_static_local: pd.DataFrame,
    params: dict,
    ROLL_START: pd.Timestamp,
    equity_start: float = 1000.0,
    max_leverage_local: float = 10.0,
    fee_entry_local: float = 0.0004,
    fee_exit_local: float = 0.0005,
    slip_entry_local: float = 0.0003,
    slip_exit_local: float = 0.0005,
    step_hours_local: int = 72,
    window_size_local: int = 18,
    topN_local: int = 5,
    exd_local: int = 10,
    hist_start_static_local: pd.Timestamp = pd.Timestamp("2025-01-01 00:00:00"),
    sim_engine_local: str = "DTW",
    A_sl_local: float = None,
    A_tp_local: float = None,
    C_sl_local: float = None,
    C_tp_local: float = None,
    sim_gate_base_local: float = None,
    ENTRY_DELAY_HOURS_local: int = None,
):
    """
    params: dict from tuner.sample_params (k_sl_A, k_tp_A, k_sl_C, k_tp_C, sim_gate, delay_h)
    returns: df_log (DataFrame) with trade rows (same schema as app의 BT-상승)
    """
    # build rolling base
    df_roll = df_full_static_local[df_full_static_local["timestamp"] >= (ROLL_START - pd.Timedelta(hours=72))].reset_index(drop=True)
    if len(df_roll) < window_size_local:
        return pd.DataFrame([])

    blocks_all = enumerate_blocks(df_roll, step_hours=step_hours_local, window_size=window_size_local)

    # find start index
    start_idx = None
    for i in range(1, len(blocks_all)):
        if blocks_all[i]["start"] >= ROLL_START:
            start_idx = i
            break
    if start_idx is None:
        return pd.DataFrame([])

    # override strat sltps by params (per-request copy)
    strat_local = {
        "A": {"method": "ATR", "k_sl": float(params.get("k_sl_A", A_sl_local or STRAT_SLTPS["A"]["k_sl"])), "k_tp": float(params.get("k_tp_A", A_tp_local or STRAT_SLTPS["A"]["k_tp"]))},
        "B": {"method": "ATR", "k_sl": float(params.get("k_sl_A", A_sl_local or STRAT_SLTPS["A"]["k_sl"])), "k_tp": float(params.get("k_tp_A", A_tp_local or STRAT_SLTPS["A"]["k_tp"]))},
        "C": {"method": "ATR", "k_sl": float(params.get("k_sl_C", C_sl_local or STRAT_SLTPS["C"]["k_sl"])), "k_tp": float(params.get("k_tp_C", C_tp_local or STRAT_SLTPS["C"]["k_tp"]))},
        "C′": {"method": "ATR", "k_sl": float(params.get("k_sl_C", C_sl_local or STRAT_SLTPS["C"]["k_sl"])), "k_tp": float(params.get("k_tp_C", C_tp_local or STRAT_SLTPS["C"]["k_tp"]))},
        "E": {"method": "ATR", "k_sl": None, "k_tp": None},
    }

    sim_gate_local = float(params.get("sim_gate", sim_gate_base_local or sim_gate_base))
    ENTRY_DELAY_HOURS_eff = int(params.get("delay_h", ENTRY_DELAY_HOURS_local or ENTRY_DELAY_HOURS))

    trade_logs = []
    eq_run = float(equity_start)

    # iterate through rolling preds
    for bp_index in range(start_idx, len(blocks_all)):
        ref_b = blocks_all[bp_index - 1]
        pred_b = blocks_all[bp_index]

        # build hist df (from hist_start_static_local)
        df_hist = df_full_static_local[df_full_static_local["timestamp"] >= hist_start_static_local].reset_index(drop=True)
        cands = get_candidates(df_hist, (ref_b["start"], ref_b["end"]), ex_margin_days=exd_local, topN=topN_local, past_only=True)
        if not cands:
            continue

        stepTD = pd.Timedelta(hours=step_hours_local)
        results = []
        for f in cands:
            next_start = f["end"]; next_end = next_start + stepTD
            df_next = df_hist[(df_hist["timestamp"] >= next_start) & (df_hist["timestamp"] < next_end)]
            if len(df_next) < window_size_local:
                continue
            closes = df_next["close"].to_numpy()
            baseC = float(closes[0])
            pct_c = (closes - baseC) / baseC * 100.0
            results.append({"sim": f["sim"], "next_start": next_start, "next_end": next_end, "pct": pct_c, "df_next": df_next.reset_index(drop=True), "base_close": baseC})
        if not results:
            continue

        # compute t_entry
        t_entry = pred_b["start"] + pd.Timedelta(hours=ENTRY_DELAY_HOURS_eff)
        if t_entry > pred_b["end"]:
            continue

        # pred segment up to t_entry
        pred_seg = df_roll[(df_roll["timestamp"] >= pred_b["start"]) & (df_roll["timestamp"] <= t_entry)]
        if len(pred_seg) == 0:
            continue

        base_cur = float(pred_seg["close"].iloc[0])
        a = ((pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)
        L = len(a)

        # find best historical flow by cosine
        best = None
        for r in results:
            b = np.array(r["pct"], dtype=float)[:L]
            sim_shape = 1.0 if (np.allclose(a, 0) and np.allclose(b, 0)) else float(cosine_similarity([a], [b])[0][0])
            if (best is None) or (sim_shape > best["sim"]):
                best = {"sim": sim_shape, "flow": r}
        if best is None:
            continue

        hist_full = np.array(best["flow"]["pct"], dtype=float)
        base_now = float(hist_full[L - 1]) if len(hist_full) > 0 else 0.0
        fut = hist_full[L - 1:] - base_now if len(hist_full) > L-1 else np.array([])
        idx_max = int(np.argmax(fut)) if fut.size > 0 else 0
        idx_min = int(np.argmin(fut)) if fut.size > 0 else 0
        max_up = float(np.max(fut)) if fut.size > 0 else 0.0
        min_dn = float(np.min(fut)) if fut.size > 0 else 0.0

        # regime & side decision
        ext_start = pred_b["start"] - pd.Timedelta(hours=48)
        prefix_end = min(t_entry, pred_b["end"])
        ext_seg = df_roll[(df_roll["timestamp"] >= ext_start) & (df_roll["timestamp"] <= prefix_end)].reset_index(drop=True)
        used_ext = (len(ext_seg) >= 2)
        seg = ext_seg if used_ext else pred_seg
        anchor = float(seg["close"].iloc[0]); last = float(seg["close"].iloc[-1])
        ret_pct = (last / anchor - 1.0) * 100.0
        cutoff = -1.0 if used_ext else 0.0
        regime_down = (ret_pct < cutoff)

        # determine preliminary side using sim_gate_local and thresholds
        side = "HOLD"
        if best["sim"] >= sim_gate_local:
            mag_up = abs(max_up); mag_dn = abs(min_dn); mag = max(mag_up, mag_dn)
            if mag >= LO_THR:
                if regime_down and (mag_up >= mag_dn):
                    side = "SHORT"
                else:
                    side = "LONG" if mag_up >= mag_dn else "SHORT"

        entry_time, entry_price, entry_target = (None, None, None)
        if side in ("LONG", "SHORT"):
            if max(abs(max_up), abs(min_dn)) >= HI_THR:
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
            # tag based on same rule
            mag = max(abs(max_up), abs(min_dn))
            up_win = (abs(max_up) >= abs(min_dn))
            dn_win = (abs(min_dn) > abs(max_up))
            if best["sim"] < sim_gate_local:
                tag_bt = "E"
            elif mag >= HI_THR:
                if dn_win or (up_win and regime_down):
                    tag_bt = "B"
                elif up_win and (not regime_down):
                    tag_bt = "A"
                else:
                    tag_bt = "E"
            elif LO_THR <= mag < HI_THR:
                if dn_win or (up_win and regime_down):
                    tag_bt = "C′"
                elif up_win and (not regime_down):
                    tag_bt = "C"
                else:
                    tag_bt = "E"
            else:
                tag_bt = "E"

            param = _resolve_sltp_by_tag(tag_bt, default_method=sltp_method, default_k_sl=k_sl, default_k_tp=k_tp, strat_sltps_override=strat_local)
            # ensure A/C override
            if tag_bt in ("A","B"):
                param["k_sl"] = strat_local["A"]["k_sl"]
                param["k_tp"] = strat_local["A"]["k_tp"]
            elif tag_bt in ("C","C′"):
                param["k_sl"] = strat_local["C"]["k_sl"]
                param["k_tp"] = strat_local["C"]["k_tp"]

            sl, tp = make_sl_tp(
                entry_price, side, method=param["method"], atr=atr_ref,
                sl_pct=param.get("sl_pct"), tp_pct=param.get("tp_pct"),
                k_sl=param.get("k_sl"), k_tp=param.get("k_tp"), tick_size=0.0
            )
        else:
            if side in ("LONG","SHORT"):
                side = "HOLD"

        size = 0.0
        used_lev = 0.0
        if side in ("LONG", "SHORT") and entry_time is not None and entry_price is not None and sl:
            size = float(eq_run) * float(max_leverage_local)  # 단순 레버리지 캡 노션널
            used_lev = float(max_leverage_local)

        exit_time, exit_price, gross_ret, net_ret = (None, None, None, None)
        if side in ("LONG", "SHORT") and entry_time is not None and entry_price is not None:
            exit_time, exit_price, gross_ret, net_ret = simulate_trade(
                df_roll, t_entry, pred_b["end"], side,
                entry_time, entry_price, sl, tp,
                fee_entry=fee_entry_local, fee_exit=fee_exit_local,
                slip_entry=slip_entry_local, slip_exit=slip_exit_local,
                exit_on_close=True
            )
        else:
            if side in ("LONG","SHORT"):
                side = "HOLD"

        ret_pct = (net_ret or 0.0) / 100.0
        eq_before = eq_run
        pnl_usd = (size or 0.0) * ret_pct
        eq_run = eq_run + pnl_usd
        ret_equity_pct = (pnl_usd / (eq_before if eq_before > 0 else 1.0)) * 100.0

        trade_logs.append({
            "pred_start": pred_b["start"], "pred_end": pred_b["end"], "t_entry": t_entry,
            "side": side, "sim_prefix": best["sim"], "scaler": "static",
            "entry_time": entry_time, "entry": entry_price, "entry_target": entry_target,
            "SL": sl, "TP": tp,
            "size_notional": size, "used_lev": used_lev, "cap_hit": False,
            "exit_time": exit_time, "exit": exit_price,
            "gross_ret_%": gross_ret, "net_ret_%": net_ret,
            "eq_before": eq_before, "eq_after": eq_run, "pnl_usd": pnl_usd, "ret_equity_%": ret_equity_pct,
            "skip_reason": None,
        })

    if not trade_logs:
        return pd.DataFrame([])

    df_log = pd.DataFrame(trade_logs).sort_values("pred_start").reset_index(drop=True)
    return df_log

# =========================
# BT-상승 (고정 기간)  — 사용자가 바꾸지 말라고 한 구간 유지
# =========================
if sim_mode == "BT-상승":
    st.subheader("BT-상승: 32h 지연 엔트리 · 블록당 1회 거래 백테스트 (Static only, ATR 고정)")
    topN = 5
    exd = 10
    stepTD = pd.Timedelta(hours=step_hours)
    delayTD = pd.Timedelta(hours=ENTRY_DELAY_HOURS)

    # (고정) 백테스트 시작점 — 기존 유지
    ROLL_START = pd.Timestamp("2025-01-01 00:00:00")

    # 블록 시퀀스 기준(공통) — static으로 시간축 고정
    df_roll_base = df_full_static[df_full_static["timestamp"] >= (ROLL_START - pd.Timedelta(hours=72))].reset_index(drop=True)
    if len(df_roll_base) < window_size:
        st.warning("BT-상승: 데이터 부족")
        st.stop()
    blocks_all = enumerate_blocks(df_roll_base, step_hours=step_hours, window_size=window_size)

    # find start index
    start_idx = None
    for i in range(1, len(blocks_all)):
        if blocks_all[i]["start"] >= ROLL_START:
            start_idx = i
            break
    if start_idx is None:
        st.warning("BT-상승: 2025년 이후 pred 블록 없음")
        st.stop()

    # 내부 평가 함수 (위 run_backtest_with_params와 동일 로직을 한 번 더 쓰지 않도록, 아래를 직접 호출)
    params_for_run = {
        "k_sl_A": float(STRAT_SLTPS["A"]["k_sl"]),
        "k_tp_A": float(STRAT_SLTPS["A"]["k_tp"]),
        "k_sl_C": float(STRAT_SLTPS["C"]["k_sl"]),
        "k_tp_C": float(STRAT_SLTPS["C"]["k_tp"]),
        "sim_gate": float(sim_gate_base),
        "delay_h": int(ENTRY_DELAY_HOURS),
    }

    df_log = run_backtest_with_params(
        df_full_static_local=df_full_static,
        params=params_for_run,
        ROLL_START=ROLL_START,
        equity_start=float(equity),
        max_leverage_local=float(max_leverage),
        fee_entry_local=float(fee_entry),
        fee_exit_local=float(fee_exit),
        slip_entry_local=float(slip_entry),
        slip_exit_local=float(slip_exit),
        step_hours_local=int(step_hours),
        window_size_local=int(window_size),
        topN_local=topN,
        exd_local=exd,
        hist_start_static_local=pd.Timestamp("2025-01-01 00:00:00"),  # 기존 유지
        sim_engine_local=sim_engine,
        A_sl_local=float(STRAT_SLTPS["A"]["k_sl"]),
        A_tp_local=float(STRAT_SLTPS["A"]["k_tp"]),
        C_sl_local=float(STRAT_SLTPS["C"]["k_sl"]),
        C_tp_local=float(STRAT_SLTPS["C"]["k_tp"]),
        sim_gate_base_local=float(sim_gate_base),
        ENTRY_DELAY_HOURS_local=int(ENTRY_DELAY_HOURS),
    )

    if df_log is None or df_log.empty:
        st.info("ROLLING 결과 없음")
        st.stop()

    df_show = df_log.copy()
    df_show = df_show.drop(columns=["gross_ret_%", "net_ret_%"], errors="ignore")
    df_show = df_show.rename(columns={"ret_equity_%": "ret_%(levered)"})
    cols = [
        "pred_start", "pred_end", "t_entry", "side", "sim_prefix", "scaler",
        "entry_time", "entry", "entry_target", "SL", "TP",
        "size_notional", "used_lev", "cap_hit", "pnl_usd",
        "ret_%(levered)", "eq_before", "eq_after", "exit_time", "exit"
    ]
    df_show = df_show[[c for c in cols if c in df_show.columns]]
    # ---------------퀴즈 정답표시-----------------
    st.subheader("퀴즈 정답")
    st.success(f"정답: {quiz['answer']}")
   
     # ---------------퀴즈 정답표시-----------------
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
            ax.set_title("Equity Curve (net) — BT-상승")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("에쿼티 커브를 그릴 수 없습니다.")
    else:
        st.info("아직 거래 로그가 없습니다. (조건 미충족/HOLD 등)")

# =========================
# 6M-상승 (최근 6개월 백테스트 & BT-상승과 비교)
# =========================
if sim_mode == "6M-상승":
    st.subheader("6M-상승: 최근 6개월 백테스트 (BT-상승과 비교)")

    # 최근 6개월 시작점 계산
    last_ts = df_full_static["timestamp"].iloc[-1]
    six_months_ago = last_ts - pd.Timedelta(days=180)
    # 4h 그리드에 맞춰 시작점 이후 첫 블록 시작으로 잡히게만 하면 됨(엄밀 정렬은 enumerate_blocks에서 처리)
    ROLL_START_6M = pd.Timestamp(six_months_ago.floor('4H'))

    params_for_run = {
        "k_sl_A": float(STRAT_SLTPS["A"]["k_sl"]),
        "k_tp_A": float(STRAT_SLTPS["A"]["k_tp"]),
        "k_sl_C": float(STRAT_SLTPS["C"]["k_sl"]),
        "k_tp_C": float(STRAT_SLTPS["C"]["k_tp"]),
        "sim_gate": float(sim_gate_base),
        "delay_h": int(ENTRY_DELAY_HOURS),
    }

    # 최근 6개월 백테스트 실행
    df_log_6m = run_backtest_with_params(
        df_full_static_local=df_full_static,
        params=params_for_run,
        ROLL_START=ROLL_START_6M,
        equity_start=float(equity),
        max_leverage_local=float(max_leverage),
        fee_entry_local=float(fee_entry),
        fee_exit_local=float(fee_exit),
        slip_entry_local=float(slip_entry),
        slip_exit_local=float(slip_exit),
        step_hours_local=int(step_hours),
        window_size_local=int(window_size),
        topN_local=5,
        exd_local=10,
        hist_start_static_local=ROLL_START_6M,  # 후보 탐색도 최근 6개월 기준
        sim_engine_local=sim_engine,
        A_sl_local=float(STRAT_SLTPS["A"]["k_sl"]),
        A_tp_local=float(STRAT_SLTPS["A"]["k_tp"]),
        C_sl_local=float(STRAT_SLTPS["C"]["k_sl"]),
        C_tp_local=float(STRAT_SLTPS["C"]["k_tp"]),
        sim_gate_base_local=float(sim_gate_base),
        ENTRY_DELAY_HOURS_local=int(ENTRY_DELAY_HOURS),
    )

    if df_log_6m is None or df_log_6m.empty:
        st.info("최근 6개월 백테스트 결과 없음 (거래 미발생/HOLD 등)")
    else:
        df_show6 = df_log_6m.copy()
        df_show6 = df_show6.drop(columns=["gross_ret_%", "net_ret_%"], errors="ignore")
        df_show6 = df_show6.rename(columns={"ret_equity_%": "ret_%(levered)"})
        cols6 = [
            "pred_start", "pred_end", "t_entry", "side", "sim_prefix", "scaler",
            "entry_time", "entry", "entry_target", "SL", "TP",
            "size_notional", "used_lev", "cap_hit", "pnl_usd",
            "ret_%(levered)", "eq_before", "eq_after", "exit_time", "exit"
        ]
        df_show6 = df_show6[[c for c in cols6 if c in df_show6.columns]]
        st.markdown("### 최근 6개월 결과 테이블")
        st.dataframe(df_show6, use_container_width=True)

        dates6, equity_curve6 = build_equity_curve(df_log_6m, float(equity))
        metrics6 = calc_metrics(df_log_6m, equity_curve6)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("거래수(6M)", metrics6["n_trades"])
        col2.metric("Hit-rate(6M)", f"{metrics6['hit_rate']:.1f}%")
        col3.metric("Avg Win/Loss(6M)", f"{metrics6['avg_win']:.2f}% / {metrics6['avg_loss']:.2f}%")
        col4.metric("Sharpe(6M, 연율화)", f"{metrics6['sharpe']:.2f}")
        col5.metric("MDD/MAR(6M)", f"{metrics6['mdd']*100:.2f}% / {metrics6['mar']:.2f}")

        if dates6 and equity_curve6 and (len(dates6) == len(equity_curve6)):
            fig, ax = plt.subplots(figsize=(10, 3.2))
            ax.plot(dates6, equity_curve6, linewidth=2)
            ax.set_title("Equity Curve (net) — 최근 6개월")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # BT-상승과 비교(요약 메트릭만)
    st.markdown("### 📊 BT-상승 vs 6M-상승 비교 요약")
    # BT-상승은 고정 시작점 유지
    ROLL_START_BT = pd.Timestamp("2025-01-01 00:00:00")
    df_log_bt = run_backtest_with_params(
        df_full_static_local=df_full_static,
        params=params_for_run,
        ROLL_START=ROLL_START_BT,
        equity_start=float(equity),
        max_leverage_local=float(max_leverage),
        fee_entry_local=float(fee_entry),
        fee_exit_local=float(fee_exit),
        slip_entry_local=float(slip_entry),
        slip_exit_local=float(slip_exit),
        step_hours_local=int(step_hours),
        window_size_local=int(window_size),
        topN_local=5,
        exd_local=10,
        hist_start_static_local=pd.Timestamp("2025-01-01 00:00:00"),
        sim_engine_local=sim_engine,
        A_sl_local=float(STRAT_SLTPS["A"]["k_sl"]),
        A_tp_local=float(STRAT_SLTPS["A"]["k_tp"]),
        C_sl_local=float(STRAT_SLTPS["C"]["k_sl"]),
        C_tp_local=float(STRAT_SLTPS["C"]["k_tp"]),
        sim_gate_base_local=float(sim_gate_base),
        ENTRY_DELAY_HOURS_local=int(ENTRY_DELAY_HOURS),
    )
if sim_mode == "오늘의 운세":
    # ======================
    # 🔮 별자리 계산 함수
    # ======================
    def get_zodiac(month, day):
        zodiac_dates = [
            ((1, 20), "물병자리"),
            ((2, 19), "물고기자리"),
            ((3, 21), "양자리"),
            ((4, 20), "황소자리"),
            ((5, 21), "쌍둥이자리"),
            ((6, 22), "게자리"),
            ((7, 23), "사자자리"),
            ((8, 23), "처녀자리"),
            ((9, 23), "천칭자리"),
            ((10, 23), "전갈자리"),
            ((11, 23), "사수자리"),
            ((12, 25), "염소자리")
        ]

        # 기본값 (12월 25일 이후 or 1월 1일 이전은 염소자리)
        for (m, d), sign in zodiac_dates:
            if (month, day) >= (m, d):
                return sign
        return "염소자리"

    # ======================
    # 🐲 띠 계산 함수
    # ======================
    def get_chinese_zodiac(year):
        animals = ["원숭이", "닭", "개", "돼지", "쥐", "소", 
                "호랑이", "토끼", "용", "뱀", "말", "양"]
        return animals[year % 12]

    # ======================
    # 🌙 Streamlit UI
    # ======================
    st.title("🔮 오늘의 운세 자동 연결")
    st.write("생년월일을 입력하면 당신의 별자리와 띠를 자동으로 계산해 운세 링크를 보여드려요!")

    birth_date = st.date_input("🎂 생년월일 입력", value=date(2000, 1, 1))

    if st.button("오늘의 운세 보기"):
        year = birth_date.year
        month = birth_date.month
        day = birth_date.day

        # 별자리 및 띠 계산
        zodiac = get_zodiac(month, day)
        animal = get_chinese_zodiac(year)

        st.subheader("✨ 당신의 정보")
        st.write(f"📅 생일: {birth_date}")
        st.write(f"🌠 별자리: **{zodiac}**")
        st.write(f"🐉 띠: **{animal}띠**")

        # 별자리 기반 링크 (예시: 네이버 운세)
        fortune_link = f"https://search.naver.com/search.naver?query={zodiac}+오늘의+운세"

        st.markdown(f"👉 [네이버에서 {zodiac} 오늘의 운세 보기]({fortune_link})", unsafe_allow_html=True)

        # 띠 기반 운세 링크도 함께 제공
        fortune_link2 = f"https://search.naver.com/search.naver?query={animal}띠+오늘의+운세"
        st.markdown(f"👉 [네이버에서 {animal}띠 오늘의 운세 보기]({fortune_link2})", unsafe_allow_html=True)

    st.caption("💡 참고: 네이버 검색 결과를 기반으로 별자리/띠별 운세 페이지로 이동합니다.")

# =========================
# 튜너 섹션: 학습은 '최근 6개월'로 고정, 끝나면 자동 저장
# =========================
st.divider()
st.header("🔧 최적 파라미터")

with st.expander("학습 실행 (최근 6개월 고정)", expanded=False):
    n_trials = st.slider("시도 횟수 (trials)", 10, 200, 40, 10)
    seed = st.number_input("Random Seed", 0, 9999, 42)

    # 튜너가 호출할 evaluate_wrapper는 최근 6개월로 고정
    def evaluate_wrapper(params: dict) -> float:
        """
        튜너용 평가 함수: 최근 6개월 ROLL_START 기준으로 run_backtest_with_params 실행.
        점수는 final equity (높을수록 좋음). 거래 없음이면 0.0
        """
        try:
            # 최근 6개월 구간 계산
            last_ts_local = df_full_static["timestamp"].iloc[-1]
            roll_start_train = pd.Timestamp((last_ts_local - pd.Timedelta(days=180)).floor('4H'))

            df_log_local = run_backtest_with_params(
                df_full_static_local=df_full_static,
                params=params,
                ROLL_START=roll_start_train,
                equity_start=float(equity),
                max_leverage_local=float(max_leverage),
                fee_entry_local=float(fee_entry),
                fee_exit_local=float(fee_exit),
                slip_entry_local=float(slip_entry),
                slip_exit_local=float(slip_exit),
                step_hours_local=int(step_hours),
                window_size_local=int(window_size),
                topN_local=5,
                exd_local=10,
                hist_start_static_local=roll_start_train,  # 후보 탐색도 6개월로 제한
                sim_engine_local=sim_engine,
                A_sl_local=float(STRAT_SLTPS["A"]["k_sl"]),
                A_tp_local=float(STRAT_SLTPS["A"]["k_tp"]),
                C_sl_local=float(STRAT_SLTPS["C"]["k_sl"]),
                C_tp_local=float(STRAT_SLTPS["C"]["k_tp"]),
                sim_gate_base_local=float(sim_gate_base),
                ENTRY_DELAY_HOURS_local=int(ENTRY_DELAY_HOURS),
            )

            if df_log_local is None or df_log_local.empty:
                return 0.0
            final_eq = float(df_log_local["eq_after"].iloc[-1])
            return float(final_eq)
        except Exception as e:
            print("evaluate_wrapper error:", e)
            return 0.0

    if st.button("🚀 튜닝 시작"):
        # run tuner
        _params = dict(
            n_trials=int(n_trials),
            n_init=8,
            N_pool=3000,
            topk=2,
            random_seed=int(seed),
            verbose=True,
            log_path=None
        )
        sig = inspect.signature(run_bayes_opt)
        allowed = {k: v for k, v in _params.items() if k in sig.parameters}

        best, df_logs = run_bayes_opt(evaluate_wrapper, **allowed)

        # 화면 출력
        st.success(f"Best score(final equity): {best['score']:.3f}")
        st.json(best["params"])
        st.dataframe(df_logs, use_container_width=True)

        # ✅ 자동 저장: 세션 & 파일
        st.session_state["tuned_params"] = best["params"]
        st.session_state["last_best_params"] = best["params"]

        try:
            save_path = Path(__file__).parent / "tuned_params.json"
        except NameError:
            # __file__이 없을 수 있는 환경(예: Streamlit Cloud) 대비
            save_path = Path("tuned_params.json")

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(best["params"], f, ensure_ascii=False, indent=2)
            st.toast(f"최적 파라미터 자동 저장 완료: {save_path}", icon="✅")
            st.caption("상단 '🧠 튜닝값 사용' 토글을 켜면 전역에 즉시 반영됩니다.")
        except Exception as e:
            st.warning(f"자동 저장 실패: {e}")


