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
from tuner import run_bayes_opt
from extra import make_ddonly_fig

# ---------------------------
# 기본 UI 설정
# ---------------------------
st.set_page_config(page_title="BTC 패턴매칭 전략 스튜디오", page_icon="📊", layout="wide")
st.title(" BTC 패턴매칭 전략 스튜디오")

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
        ["현재구간", "백테스트", "백테스트1", "튜닝·운세", "백테스트뷰"],
        index=0,
        horizontal=True,
        help="현재구간: 단일·32h / 백테스트: 고정 시계열 백테스트 / 백테스트1: 최근 6개월 백테스트 / 튜닝·운세: 시장상황을 간단 지표로 보여주고 최적 파라미터 계산 / 백테스트뷰: 과거 거래 흐름을 시각적으로 확인"
    )

sim_engine = "DTW"
w_dtw = 0.5

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
        base = Path(".")
    p = base / "tuned_params.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                params = json.load(f)
            st.session_state.setdefault("tuned_params", params)
            st.session_state.setdefault("last_best_params", params)
            st.session_state["tuned_params_path"] = str(p)
            return True
        except Exception as e:
            st.warning(f"tuned_params.json 로드 실패: {e}")
    return False


_loaded_tp = _load_tuned_params_into_session()
# ---------------------------
# 속도 개선
# ---------------------------
@st.cache_data(show_spinner=False)
def load_all_data():
    client = connect_binance()
    df_raw = fetch_futures_4h_klines(client, start_time="2020-01-01")
    df_funding = fetch_funding_rate(client, start_time="2020-01-01")
    df_feat = add_features(df_raw, df_funding)

    train_end_ts_static = pd.Timestamp("2022-07-01 00:00:00")
    df_full_static = apply_static_zscore(df_feat, GLOBAL_Z_COLS, train_end_ts_static)
    df_full_static = finalize_preprocessed(df_full_static, window_size)
    return df_full_static

@st.cache_data(show_spinner=False)
def cached_blocks(df, step_hours, window_size):
    return enumerate_blocks(df, step_hours=step_hours, window_size=window_size)

@st.cache_data(show_spinner=False)
def cached_vectors(df, step_hours, window_size):
    blocks = enumerate_blocks(df, step_hours=step_hours, window_size=window_size)
    out = []
    for b in blocks:
        w = df[(df["timestamp"] >= b["start"]) & (df["timestamp"] < b["end"])]
        if len(w) >= window_size:
            v = window_vector(w.iloc[:window_size], L=window_size)
            out.append((b, v))
    return out
# ---------------------------
# 튜닝값 사용 토글 + 전역 주입
# ---------------------------
use_tuned = st.toggle(
    "튜닝값 사용",
    value=bool(st.session_state.get("tuned_params")),
    help="튜닝 섹션에서 저장된 best params를 NOW/BT-상승/6M-상승에 주입"
)
tuned = st.session_state.get("tuned_params")
if use_tuned and tuned:
    sim_gate_base = float(tuned.get("sim_gate", sim_gate_base))
    ENTRY_DELAY_HOURS = float(tuned.get("delay_h", ENTRY_DELAY_HOURS))

    STRAT_SLTPS["A"]["k_sl"] = STRAT_SLTPS["B"]["k_sl"] = float(tuned.get("k_sl_A", STRAT_SLTPS["A"]["k_sl"]))
    STRAT_SLTPS["A"]["k_tp"] = STRAT_SLTPS["B"]["k_tp"] = float(tuned.get("k_tp_A", STRAT_SLTPS["A"]["k_tp"]))
    STRAT_SLTPS["C"]["k_sl"] = STRAT_SLTPS["C′"]["k_sl"] = float(tuned.get("k_sl_C", STRAT_SLTPS["C"]["k_sl"]))
    STRAT_SLTPS["C"]["k_tp"] = STRAT_SLTPS["C′"]["k_tp"] = float(tuned.get("k_tp_C", STRAT_SLTPS["C"]["k_tp"]))

    st.caption(
        f"sim_gate={sim_gate_base:.3f}, delay_h={ENTRY_DELAY_HOURS:.0f}, "
        f"A/B k_sl={STRAT_SLTPS['A']['k_sl']:.2f}, k_tp={STRAT_SLTPS['A']['k_tp']:.2f}, "
        f"C/C′ k_sl={STRAT_SLTPS['C']['k_sl']:.2f}, k_tp={STRAT_SLTPS['C']['k_tp']:.2f}"
    )
# ---------------------------
# 현재구간 전용 SL/TP 입력 (튜닝값 미사용 시)
# ---------------------------
if (sim_mode == "현재구간") and (not use_tuned):
    colA_now, colB_now = st.columns(2)

    with colA_now:
        A_sl = st.number_input(
            "A/B SL(×ATR)",
            0.1, 50.0,
            float(STRAT_SLTPS["A"]["k_sl"]),
            0.1,
            help="현재구간에서 A/B 전략의 손절 배수(ATR 기준)"
        )
        A_tp = st.number_input(
            "A/B TP(×ATR)",
            0.1, 50.0,
            float(STRAT_SLTPS["A"]["k_tp"]),
            0.1,
            help="현재구간에서 A/B 전략의 익절 배수(ATR 기준)"
        )

    with colB_now:
        C_sl = st.number_input(
            "C/C′ SL(×ATR)",
            0.1, 50.0,
            float(STRAT_SLTPS["C"]["k_sl"]),
            0.1,
            help="현재구간에서 C/C′ 전략의 손절 배수(ATR 기준)"
        )
        C_tp = st.number_input(
            "C/C′ TP(×ATR)",
            0.1, 50.0,
            float(STRAT_SLTPS["C"]["k_tp"]),
            0.1,
            help="현재구간에서 C/C′ 전략의 익절 배수(ATR 기준)"
        )

    # 입력값을 전역 STRAT_SLTPS에 반영
    STRAT_SLTPS["A"]["k_sl"] = STRAT_SLTPS["B"]["k_sl"] = float(A_sl)
    STRAT_SLTPS["A"]["k_tp"] = STRAT_SLTPS["B"]["k_tp"] = float(A_tp)
    STRAT_SLTPS["C"]["k_sl"] = STRAT_SLTPS["C′"]["k_sl"] = float(C_sl)
    STRAT_SLTPS["C"]["k_tp"] = STRAT_SLTPS["C′"]["k_tp"] = float(C_tp)
# ---------------------------
# 백테스트/백테스트1 공통 UI (수수료/SLTP 입력)
# ---------------------------
if sim_mode in ("백테스트", "백테스트1"):
    colA, colB, colC = st.columns(3)
    with colA:
        sim_engine = st.selectbox("유사도 방식", ["DTW", "Cosine"], index=0, help="DTW: 시간 변형을 허용한 패턴 유사도 / Cosine: 방향성 중심 유사도")
        A_sl = st.number_input("A/B SL(×ATR)", 0.1, 50.0, 1.0, 0.1,help="A/B 전략 손절 폭: ATR의 몇 배까지 역행하면 손절할지 설정")
        A_tp = st.number_input("A/B TP(×ATR)", 0.1, 50.0, 2.5, 0.1,help="A/B 전략 익절 폭: ATR의 몇 배 수익에서 익절할지 설정")
        C_sl = st.number_input("C/C′ SL(×ATR)", 0.1, 50.0, 1.5, 0.1,help="C/C′ 전략 손절 폭: ATR의 몇 배까지 역행하면 손절할지 설정")
        C_tp = st.number_input("C/C′ TP(×ATR)", 0.1, 50.0, 1.5, 0.1,help="C/C′ 전략 익절 폭: ATR의 몇 배 수익에서 익절할지 설정")

    if not use_tuned:
        STRAT_SLTPS["A"]["k_sl"] = STRAT_SLTPS["B"]["k_sl"] = float(A_sl)
        STRAT_SLTPS["A"]["k_tp"] = STRAT_SLTPS["B"]["k_tp"] = float(A_tp)
        STRAT_SLTPS["C"]["k_sl"] = STRAT_SLTPS["C′"]["k_sl"] = float(C_sl)
        STRAT_SLTPS["C"]["k_tp"] = STRAT_SLTPS["C′"]["k_tp"] = float(C_tp)

    # 2번째 열: 수수료/슬리피지
    with colB:
        fee_entry = st.number_input("진입 수수료(%)", 0.0, 1.0, 0.04, 0.01,help="롱/숏 포지션을 새로 잡을 때 거래소에 내는 수수료") / 100.0
        fee_exit  = st.number_input("청산 수수료(%)", 0.0, 1.0, 0.05, 0.01,help="포지션을 종료(청산)할 때 거래소에 내는 수수료") / 100.0
        slip_entry = st.number_input("진입 슬리피지(%)", 0.0, 0.5, 0.03, 0.01,help="주문 가격보다 불리한 가격에 진입될 수 있는 오차(미끄러짐)") / 100.0
        slip_exit  = st.number_input("청산 슬리피지(%)", 0.0, 0.5, 0.05, 0.01,help="예상 청산 가격보다 불리한 가격으로 체결될 수 있는 오차") / 100.0

    # 3번째 열: 가상 Equity/레버리지
    with colC:
        equity = st.number_input("가상 Equity (USDT)", 10.0, value=1000.0, step=10.0,help="백테스트·시뮬레이션에 사용할 초기 자본(가상의 계좌 잔고)")
        max_leverage = st.number_input("최대 레버리지(x)", 1.0, 50.0, 10.0, 1.0,help="포지션을 잡을 때 적용할 최대 배율로, 포지션 크기 계산에 사용")

# ---------------------------
# 데이터 로드 & 전처리
# ---------------------------

df_full_static = load_all_data()

if len(df_full_static) < window_size:
    st.error("데이터가 부족합니다.")
    st.stop()

now_ts = df_full_static["timestamp"].iloc[-1]
(ref_start, ref_end), (pred_start, pred_end) = pick_blocks(now_ts, step_hours=step_hours)

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
    blocks_cached = cached_vectors(df, step_hours, window_size)  
    ex_margin = pd.Timedelta(days=ex_margin_days)
    F = len(FEAT_COLS)
    cand = []
    for b, vec_hist in blocks_cached:
        if past_only:
            if not (b["end"] <= ref_range[0] - ex_margin):
                continue
        else:
            if not ((b["end"] <= ref_range[0] - ex_margin) or (b["start"] >= ref_range[1] + ex_margin)):
                continue
        sim = sim_tier3(vec_ref, vec_hist, L=window_size, F=F,
                        mode=sim_engine, w_dtw=w_dtw)

        cand.append({
            "start": b["start"],
            "end": b["end"],
            "sim": sim
        })
    cand.sort(key=lambda x: x["sim"], reverse=True)
    return cand[:topN]


def _adjust_magnitude(pct_mag: float) -> float:
    return max(0.0, pct_mag - 0.1)


def _get_close_at_or_before(df: pd.DataFrame, ts: pd.Timestamp):
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
# 현재구간
# =========================
if sim_mode == "현재구간":
    df_full = df_full_static[df_full_static["timestamp"] >= pd.Timestamp("2025-01-01 00:00:00")].reset_index(drop=True)

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
        "base_close",
        best["flow"].get(
            "base_close_28h",
            (float(df_best_next["close"].iloc[L - 1]) if len(df_best_next) >= L and L > 0 else float(df_best_next["close"].iloc[-1]))
        )
    )
    st.markdown("###  시간 정보")
    st.write({"현재 블록 구간": f"{pred_start} ~ {pred_end}"})

    fig, ax = plt.subplots(figsize=(9, 3))
    hist_full = np.array(best["flow"]["pct"], dtype=float)
    ax.plot(np.arange(len(hist_full)), hist_full, label="매칭 72h(원시%)")
    ax.plot(np.arange(len(a_plot)), a_plot, label=f"현재 진행 (L={len(a_plot)})")
    ax.axvline(L - 1, ls="--", label="엔트리 기준(32h)")
    ax.axhline(HI_THR, ls="--"); ax.axhline(-HI_THR, ls="--")
    ax.axhline(LO_THR, ls=":"); ax.axhline(-LO_THR, ls=":")
    ax.axhline(0, ls=":")
    ax.set_title("현재구간: 32h 기준 · 진행 vs 매칭 ")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    fut = hist_full[min(int(np.ceil(ENTRY_DELAY_HOURS / 4.0)), len(hist_full) - 1):] - hist_full[min(int(np.ceil(ENTRY_DELAY_HOURS / 4.0)), len(hist_full) - 1)]
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

    st.markdown(f"###  현재 판정: **{current_scenario} 시나리오**")
    st.caption(f"현재 유사도 = {best['sim']:.3f} / 게이트 = {sim_gate_base:.2f}")
    st.write(f" 현재 데이터 최신 시점: {now_ts}")

    STRAT_DESC = {
        "A": "강한 상승: HI_THR 이상 & 상승우위 & 비하락레짐 → 다음봉 시가 LONG",
        "B": "강한 하락: HI_THR 이상 & (하락우위 또는 하락레짐+상승우위) → 다음봉 시가 SHORT",
        "C": "중간 상승: LO~HI & 상승우위 & 비하락레짐 → 되돌림 리밋가 LONG",
        "C′": "중간 하락: LO~HI & (하락우위 또는 하락레짐+상승우위) → 되돌림 리밋가 SHORT",
        "E": "약함/유사도 미달 → HOLD"
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

        if tag in ("A", "B"):
            entry_price = float(ENTRY_FIX_PRICE)
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
                else:
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

    cond_A = (mag >= HI_THR) and up_win and (not regime_down)
    cond_B = (mag >= HI_THR) and dn_win
    cond_C = (LO_THR <= mag < HI_THR) and up_win and (not regime_down)
    cond_Cp = (LO_THR <= mag < HI_THR) and dn_win
    cond_E = (mag < LO_THR) or (best["sim"] < sim_gate)

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

        df_scn[["SL_Δ", "TP_Δ", "SL_%", "TP_%"]] = df_scn.apply(_delta, axis=1)
        show_cols = ["scenario", "설명", "side", "entry_price", "SL", "TP", "SL_Δ", "TP_Δ", "SL_%", "TP_%", "cond_ok", "min_entry_time", "note"]
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

    df_roll = df_full_static_local[df_full_static_local["timestamp"] >= (ROLL_START - pd.Timedelta(hours=72))].reset_index(drop=True)
    if len(df_roll) < window_size_local:
        return pd.DataFrame([])

    blocks_all = cached_blocks(df_roll, step_hours, window_size)

    start_idx = None
    for i in range(1, len(blocks_all)):
        if blocks_all[i]["start"] >= ROLL_START:
            start_idx = i
            break
    if start_idx is None:
        return pd.DataFrame([])

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

    for bp_index in range(start_idx, len(blocks_all)):
        ref_b = blocks_all[bp_index - 1]
        pred_b = blocks_all[bp_index]

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

        t_entry = pred_b["start"] + pd.Timedelta(hours=ENTRY_DELAY_HOURS_eff)
        if t_entry > pred_b["end"]:
            continue

        pred_seg = df_roll[(df_roll["timestamp"] >= pred_b["start"]) & (df_roll["timestamp"] <= t_entry)]
        if len(pred_seg) == 0:
            continue

        base_cur = float(pred_seg["close"].iloc[0])
        a = ((pred_seg["close"] - base_cur) / base_cur * 100.0).to_numpy(dtype=float)
        L = len(a)

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
        fut = hist_full[L - 1:] - base_now if len(hist_full) > L - 1 else np.array([])
        idx_max = int(np.argmax(fut)) if fut.size > 0 else 0
        idx_min = int(np.argmin(fut)) if fut.size > 0 else 0
        max_up = float(np.max(fut)) if fut.size > 0 else 0.0
        min_dn = float(np.min(fut)) if fut.size > 0 else 0.0

        ext_start = pred_b["start"] - pd.Timedelta(hours=48)
        prefix_end = min(t_entry, pred_b["end"])
        ext_seg = df_roll[(df_roll["timestamp"] >= ext_start) & (df_roll["timestamp"] <= prefix_end)].reset_index(drop=True)
        used_ext = (len(ext_seg) >= 2)
        seg = ext_seg if used_ext else pred_seg
        anchor = float(seg["close"].iloc[0]); last = float(seg["close"].iloc[-1])
        ret_pct = (last / anchor - 1.0) * 100.0
        cutoff = -1.0 if used_ext else 0.0
        regime_down = (ret_pct < cutoff)

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
        tag_bt = "E"

        sl, tp = (None, None)
        if side in ("LONG", "SHORT") and entry_time is not None and entry_price is not None:
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
            if tag_bt in ("A", "B"):
                param["k_sl"] = strat_local["A"]["k_sl"]
                param["k_tp"] = strat_local["A"]["k_tp"]
            elif tag_bt in ("C", "C′"):
                param["k_sl"] = strat_local["C"]["k_sl"]
                param["k_tp"] = strat_local["C"]["k_tp"]

            sl, tp = make_sl_tp(
                entry_price, side, method=param["method"], atr=atr_ref,
                sl_pct=param.get("sl_pct"), tp_pct=param.get("tp_pct"),
                k_sl=param.get("k_sl"), k_tp=param.get("k_tp"), tick_size=0.0
            )
        else:
            if side in ("LONG", "SHORT"):
                side = "HOLD"

        size = 0.0
        used_lev = 0.0
        if side in ("LONG", "SHORT") and entry_time is not None and entry_price is not None and sl:
            size = float(eq_run) * float(max_leverage_local)
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
            if side in ("LONG", "SHORT"):
                side = "HOLD"

        ret_pct_equity = (net_ret or 0.0) / 100.0
        eq_before = eq_run
        pnl_usd = (size or 0.0) * ret_pct_equity
        eq_run = eq_run + pnl_usd
        ret_equity_pct = (pnl_usd / (eq_before if eq_before > 0 else 1.0)) * 100.0

        trade_logs.append({
            "pred_start": pred_b["start"], "pred_end": pred_b["end"], "t_entry": t_entry,
            "side": side, "tag": tag_bt, "sim_prefix": best["sim"], "scaler": "static",
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
# 공통: 파라미터 dict / 롤링 실행 헬퍼 (중복 제거)
# =========================
def build_params_for_run():
    return {
        "k_sl_A": float(STRAT_SLTPS["A"]["k_sl"]),
        "k_tp_A": float(STRAT_SLTPS["A"]["k_tp"]),
        "k_sl_C": float(STRAT_SLTPS["C"]["k_sl"]),
        "k_tp_C": float(STRAT_SLTPS["C"]["k_tp"]),
        "sim_gate": float(sim_gate_base),
        "delay_h": int(ENTRY_DELAY_HOURS),
    }


def run_bt_for_range(roll_start: pd.Timestamp, hist_start: pd.Timestamp, params_override: dict | None = None):
    params = params_override if params_override is not None else build_params_for_run()
    return run_backtest_with_params(
        df_full_static_local=df_full_static,
        params=params,
        ROLL_START=roll_start,
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
        hist_start_static_local=hist_start,
        sim_engine_local=sim_engine,
        A_sl_local=float(STRAT_SLTPS["A"]["k_sl"]),
        A_tp_local=float(STRAT_SLTPS["A"]["k_tp"]),
        C_sl_local=float(STRAT_SLTPS["C"]["k_sl"]),
        C_tp_local=float(STRAT_SLTPS["C"]["k_tp"]),
        sim_gate_base_local=float(sim_gate_base),
        ENTRY_DELAY_HOURS_local=int(ENTRY_DELAY_HOURS),
    )
def run_bt_solo_log():
    ROLL_START = pd.Timestamp("2025-01-01 00:00:00")

    df_log = run_bt_for_range(
        roll_start=ROLL_START,
        hist_start=ROLL_START,
        params_override=None
    )

    return df_log

def get_roll_start_6m(df: pd.DataFrame):
    last_ts_local = df["timestamp"].iloc[-1]
    return pd.Timestamp((last_ts_local - pd.Timedelta(days=180)).floor('4H'))


def show_bt_result(label_prefix: str, df_log: pd.DataFrame, base_equity: float):
    if df_log is None or df_log.empty:
        st.info(f"{label_prefix} 백테스트 결과 없음 (거래 미발생/HOLD 등)")
        return

    df_show = df_log.copy()
    df_show = df_show.drop(columns=["gross_ret_%", "net_ret_%"], errors="ignore")
    df_show = df_show.rename(columns={"ret_equity_%": "ret_%(levered)"})
    cols = [
        "pred_start","pred_end","t_entry","side","tag","sim_prefix","scaler",
        "entry_time","entry","entry_target","SL","TP",
        "size_notional","used_lev","cap_hit","pnl_usd",
        "ret_%(levered)","eq_before","eq_after","exit_time","exit"
    ]
    df_show = df_show[[c for c in cols if c in df_show.columns]]
    st.markdown(f"### {label_prefix} 결과 테이블 (레버리지 반영 수익률)")
    st.dataframe(df_show, use_container_width=True)

    dates, equity_curve = build_equity_curve(df_log, float(base_equity))
    metrics = calc_metrics(df_log, equity_curve)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(f"거래수{label_prefix}", metrics["n_trades"])
    col2.metric(f"Hit-rate{label_prefix}", f"{metrics['hit_rate']:.1f}%")
    col3.metric(f"Avg Win/Loss{label_prefix}", f"{metrics['avg_win']:.2f}% / {metrics['avg_loss']:.2f}%")
    col4.metric(f"Sharpe{label_prefix}", f"{metrics['sharpe']:.2f}")
    col5.metric(f"MDD / MAR{label_prefix}", f"{metrics['mdd']*100:.2f}% / {metrics['mar']:.2f}")

    if dates and equity_curve and (len(dates) == len(equity_curve)):
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(dates, equity_curve, linewidth=2)
        ax.set_title(f"Equity Curve (net) — {label_prefix}")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.warning(f"{label_prefix} 에쿼티 커브를 그릴 수 없습니다.")

    # --- Tag-wise summary ---
    if "tag" in df_log.columns:
        st.markdown("#### 태그별 수익")
        groups = []
        for t, g in df_log.groupby("tag"):
            d, eq = build_equity_curve(g, float(base_equity))
            final_eq = float(eq[-1]) if eq else float(base_equity)
            m = calc_metrics(g, eq)
            groups.append({
                "tag": t,
                "trades": m.get("n_trades"),
                "hit(%)": round(m.get("hit_rate", 0.0), 1),
                "avg_win(%)": round(m.get("avg_win", 0.0), 2),
                "avg_loss(%)": round(m.get("avg_loss", 0.0), 2),
                "Sharpe": round(m.get("sharpe", 0.0), 2),
                "Final Eq": round(final_eq, 2),
                "MAR": None if m.get("mar") != m.get("mar") else round(m.get("mar", 0.0), 2),
            })
        if groups:
            st.dataframe(pd.DataFrame(groups).sort_values("tag"), use_container_width=True)

# =========================
# 백테스트
# =========================
if sim_mode == "백테스트":
    ROLL_START = pd.Timestamp("2025-01-01 00:00:00")

    df_roll_base = df_full_static[df_full_static["timestamp"] >= (ROLL_START - pd.Timedelta(hours=72))].reset_index(drop=True)
    if len(df_roll_base) < window_size:
        st.warning("백테스트: 데이터 부족")
        st.stop()
    blocks_all = enumerate_blocks(df_roll_base, step_hours=step_hours, window_size=window_size)

    start_idx = None
    for i in range(1, len(blocks_all)):
        if blocks_all[i]["start"] >= ROLL_START:
            start_idx = i
            break
    if start_idx is None:
        st.warning("백테스트: 2025년 이후 pred 블록 없음")
        st.stop()

    df_log = run_bt_for_range(ROLL_START, pd.Timestamp("2025-01-01 00:00:00"))

    if df_log is None or df_log.empty:
        st.info("ROLLING 결과 없음")
        st.stop()
    show_bt_result("백테스트", df_log, equity)
    st.markdown("#### MDD 시각화")
    if "df_log" in locals() and df_log is not None and len(df_log):
        st.pyplot(make_ddonly_fig(df_log), clear_figure=True)
    else:
        st.info("표시할 트레이드 로그가 없습니다.")

# =========================
# 백테스트1
# =========================
if sim_mode == "백테스트1":
    ROLL_START_6M = get_roll_start_6m(df_full_static)

    # 이 블록 내부에서만 사용할 후보풀 시작 시점
    CAND_HIST_START = pd.Timestamp("2025-01-01 00:00:00")

    # 후보풀만 고정해서 실행 (시작자본/표시는 기존 equity 유지)
    df_log_6m = run_bt_for_range(
        roll_start=ROLL_START_6M,
        hist_start=CAND_HIST_START,
        params_override=None
    )
    show_bt_result("최근 6개월", df_log_6m, equity)

# =========================
# 오늘의 운세 + (튜너 병합)
# =========================
if sim_mode == "튜닝·운세":
    df_full = df_full_static
    now_ts = df_full["timestamp"].iloc[-1]

    # 제목은 심플하게
    st.header("운세")

    # -------------------------
    # 변동성 · 펀딩 컨디션 블록
    # -------------------------
    horizon_h = int(ENTRY_DELAY_HOURS)  # 기본 32h 구간
    cut_h = now_ts - pd.Timedelta(hours=horizon_h)
    seg_h = df_full[df_full["timestamp"] >= cut_h]

    if len(seg_h) < 2 or df_full["log_ret"].std(ddof=0) == 0:
        st.info("컨디션 점수를 계산하기 위한 데이터가 부족합니다.")
    else:
        # --- 변동성 비율 ---
        vol_recent = float(seg_h["log_ret"].std(ddof=0))
        vol_all = float(df_full["log_ret"].std(ddof=0))
        vol_ratio = vol_recent / vol_all if vol_all > 0 else 1.0

        # --- 펀딩 z-score ---
        if "funding" in df_full.columns:
            fund_recent = float(seg_h["funding"].mean())
            fund_mu = float(df_full["funding"].mean())
            fund_sd = float(df_full["funding"].std(ddof=0))
            funding_z = 0.0 if fund_sd == 0 else (fund_recent - fund_mu) / fund_sd
        else:
            funding_z = 0.0

        # --- 점수 계산: 0 ~ 100 ---
        base = 50.0 * (vol_ratio - 1.0)          # 변동성이 평소보다 높으면 +, 낮으면 -
        base = max(-50.0, min(50.0, base))
        penalty = min(30.0, 15.0 * abs(funding_z))   # 펀딩 쏠림이 크면 감점
        score = base - penalty + 50.0                # 중앙 50 기준
        score = max(0.0, min(100.0, score))

        # --- 레짐 분류 ---
        if vol_ratio <= 0.8:
            vol_regime = "저변동"
        elif vol_ratio >= 1.2:
            vol_regime = "고변동"
        else:
            vol_regime = "보통"

        abs_fz = abs(funding_z)
        if funding_z >= 0.7:
            funding_regime = "롱 과열"
        elif funding_z <= -0.7:
            funding_regime = "숏 과열"
        else:
            funding_regime = "중립"

        # --- 시장 컨디션 코멘트 (comment 느낌) ---
        if score >= 70:
            score_comment = "시장 컨디션: 양호 — 평소 수준 이상으로 공격적인 진입도 고려 가능."
        elif score >= 40:
            score_comment = "시장 컨디션: 보통 — 기본 전략대로 운용하는 구간."
        else:
            score_comment = "시장 컨디션: 불리 — 진입·레버리지를 보수적으로 가져가는 편이 안전한 구간."

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "시장 컨디션 점수",
                f"{score:.1f} / 100",
                help=(
                    f"최근 {horizon_h}시간의 변동성·펀딩 상태를 0~100으로 스케일한 값입니다. "
                    "50이 과거 평균 수준, 낮을수록 보수적인 대응이 권장되는 구간입니다."
                ),
            )
        with col2:
            st.metric(
                "변동성 레짐",
                vol_regime,
                f"{vol_ratio:.2f}x",
                help=(
                    "최근 로그수익률 표준편차 / 전체 기간 표준편차 비율입니다. "
                    "0.8 이하: 저변동, 0.8~1.2: 보통, 1.2 이상: 고변동."
                ),
            )
        with col3:
            st.metric(
                "펀딩 레짐",
                funding_regime,
                f"|z| = {abs_fz:.2f}",
                help=(
                    "최근 구간 펀딩비를 전체 평균/표준편차로 z-score화한 값입니다. "
                    "절댓값이 클수록 한쪽 포지션(롱/숏) 쏠림이 강한 상태를 의미합니다."
                ),
            )
    st.markdown(f"{score_comment}")
    st.header(" 최적 파라미터")

    with st.expander("학습 실행 (최근 6개월 고정)", expanded=False):
        n_trials = st.slider("시도 횟수 (trials)", 10, 200, 40, 10,
                             help="튜닝을 몇 회 반복할지 설정합니다. 반복 수가 많을수록 더 정교한 값 탐색.")
        seed = st.number_input("Random Seed", 0, 9999, 42,
                               help="튜닝 결과를 재현 가능하게 만드는 난수 고정값입니다.")

        # ⚠️ 외부 새 함수 안 씀. 이미 있는 df_full_static / run_backtest_with_params만 사용.
        def evaluate_wrapper(params: dict) -> float:
           
            try:
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
                    hist_start_static_local=roll_start_train,  # 후보 탐색도 최근 6개월로 제한
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

        if st.button(" 튜닝 시작"):
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

            st.success(f"Best score(final equity): {best['score']:.3f}")
            st.json(best["params"])
            st.dataframe(df_logs, use_container_width=True)

            # 세션·파일 저장 (기존 로직 유지)
            st.session_state["tuned_params"] = best["params"]
            st.session_state["last_best_params"] = best["params"]

            try:
                save_path = Path(__file__).parent / "tuned_params.json"
            except NameError:
                save_path = Path("tuned_params.json")

            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(best["params"], f, ensure_ascii=False, indent=2)
                st.toast(f"최적 파라미터 자동 저장 완료: {save_path}", icon="✅")
                st.caption("상단 ' 튜닝값 사용' 토글을 켜면 전역에 즉시 반영됩니다.")
            except Exception as e:
                st.warning(f"자동 저장 실패: {e}")
if sim_mode == "백테스트뷰":
    df_log = run_bt_solo_log()
    if df_log is None or df_log.empty:
        st.error("백테스트 거래 로그 없음")
        st.stop()

    # side가 아니라 tag로 필터
    df_real = df_log[
        df_log["tag"].isin(["A", "B", "C", "C′"]) &
        df_log["exit_time"].notna()
    ].reset_index(drop=True)

    if df_real.empty:
        st.info("리뷰할 거래 없음")
        st.stop()

    pick = st.selectbox(
        "리뷰할 거래 선택",
        options=df_real.index,
        format_func=lambda i: f"{df_real.loc[i, 't_entry']} | {df_real.loc[i, 'side']} | {df_real.loc[i, 'net_ret_%']:.2f}%",
        help="백테스트에서 실제로 실행된 거래 중 하나를 선택해 상세 구간을 확인합니다."
    )

    row = df_real.loc[pick]

    seg = df_full_static[
        (df_full_static["timestamp"] >= row["t_entry"] - pd.Timedelta(hours=24)) &
        (df_full_static["timestamp"] <= row["exit_time"] + pd.Timedelta(hours=24))
    ].reset_index(drop=True)

    base = seg["close"].iloc[0]
    pct = (seg["close"] / base - 1) * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(seg["timestamp"], pct, linewidth=2)

    ax.scatter(row["t_entry"], (row["entry"] / base - 1) * 100, s=120, color="green")
    ax.scatter(row["exit_time"], (row["exit"] / base - 1) * 100, s=120, color="red")

    st.pyplot(fig)