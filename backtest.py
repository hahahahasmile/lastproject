# backtest_1m_simple.py
# pip install python-binance pandas numpy

import os
import pandas as pd
from datetime import timedelta
from binance.client import Client

# =====================================================
# 🔧 여기 값들만 직접 수정하면 됨
# =====================================================
SYMBOL      = "BTCUSDT"
SIDE        = "LONG"          # "LONG" 또는 "SHORT"
ENTRY       = "2025-09-11 04:00:00"   # 진입 시점 (UTC로 해석)
ENTRY_PRICE = 113729         # 터치-진입 가격 (None이면 첫 분봉 시가로 즉시 진입)
TP_PRICE    = 116682         # 익절가 (필수)
SL_PRICE    = 112745        # 손절가 (없으면 None)
LEVERAGE    = 10.0
FEE_ENTRY   = 0.0004          # 0.04%
FEE_EXIT    = 0.0005          # 0.05%
SIZE        = 1000.0           # 증거금(USDT)
CSV_PATH    = "backtest_path.csv"  # None이면 저장 안 함
# =====================================================


def parse_utc(ts_like):
    """Z 없이 써도 UTC로 해석. tz 없는 경우 UTC로 로컬라이즈, 있으면 그대로 사용."""
    ts = pd.Timestamp(ts_like)
    return ts if ts.tzinfo else ts.tz_localize("UTC")


def fetch_futures_1m(client, symbol, start_utc, end_utc):
    """1분봉 조회 (≤25시간). 반환: time, open, high, low, close, volume (UTC, 오름차순)"""
    st = pd.Timestamp(start_utc)
    et = pd.Timestamp(end_utc)
    kl = client.futures_klines(
        symbol=symbol, interval="1m",
        startTime=int(st.timestamp()*1000),
        endTime=int(et.timestamp()*1000),
        limit=1500
    )
    if not kl:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])
    df = pd.DataFrame(kl, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    df = df[["open_time","open","high","low","close","volume"]].copy()
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)


def main():
    # Binance Futures 클라이언트 (키 없어도 보통 klines는 동작하지만 환경에 따라 필요)
    client = Client(os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_API_SECRET", ""))

    # 시간 범위: ENTRY ~ ENTRY+12h
    entry_ts = parse_utc(ENTRY)
    deadline = entry_ts + timedelta(hours=40)

    # 1분봉 로드 (여유 1분)
    df = fetch_futures_1m(client, SYMBOL, entry_ts - timedelta(minutes=1), deadline + timedelta(minutes=1))
    if df.empty:
        print("ERR: 1분봉 불러오기 실패")
        return

    # ENTRY 시각 이상 분봉만 사용
    df_after = df[df["time"] >= entry_ts].reset_index(drop=True)
    if df_after.empty:
        print("ERR: entry 이후 캔들이 없음")
        return

    # =========================
    # 엔트리 체결 로직 (패치)
    # =========================
    entry_filled = False
    entry_idx = 0
    entry_time = None

    if ENTRY_PRICE is None:
        # 즉시 진입: ENTRY 이상 첫 분봉 시가로 체결
        entry_idx = 0
        entry_time = df_after.iloc[0]["time"]
        entry_price = float(df_after.iloc[0]["open"])
        entry_filled = True
    else:
        # 터치-진입: low <= ENTRY_PRICE <= high 인 첫 분봉에서 체결
        want = float(ENTRY_PRICE)
        for i, r in df_after.iterrows():
            if float(r["low"]) <= want <= float(r["high"]):
                entry_idx = i
                entry_time = r["time"]
                entry_price = want
                entry_filled = True
                break

    if not entry_filled:
        print("NO_FILL: 12시간 내 ENTRY_PRICE에 닿지 않아 미체결")
        return

    # 엔트리 이후 구간만으로 시뮬레이션 진행
    df_after = df_after.iloc[entry_idx:].reset_index(drop=True)

    # 수량 (증거금 기준)
    qty = (SIZE * LEVERAGE) / entry_price
    side_mult = 1.0 if SIDE == "LONG" else -1.0

    # =========================
    # 시뮬레이션 (TP/SL → 데드라인 종가)
    # =========================
    hit, exit_price, exit_time = None, None, None
    for _, r in df_after.iterrows():
        t = r["time"]; H = float(r["high"]); L = float(r["low"]); C = float(r["close"])

        # 손절/익절 체크 (동시충족 시 SL 우선)
        hit_sl = (SL_PRICE is not None) and ((L <= SL_PRICE) if SIDE=="LONG" else (H >= SL_PRICE))
        hit_tp = (TP_PRICE is not None) and ((H >= TP_PRICE) if SIDE=="LONG" else (L <= TP_PRICE))

        if hit_sl and hit_tp:
            hit = "SL"; exit_price = float(SL_PRICE); exit_time = t; break
        elif hit_sl:
            hit = "SL"; exit_price = float(SL_PRICE); exit_time = t; break
        elif hit_tp:
            hit = "TP"; exit_price = float(TP_PRICE); exit_time = t; break

        if t >= deadline:
            hit = "DEADLINE"; exit_price = C; exit_time = t; break

    if hit is None:
        last = df_after.iloc[-1]
        hit = "DEADLINE"; exit_price = float(last["close"]); exit_time = last["time"]

    # 손익 계산 (증거금 대비)
    pnl_usdt = qty * (exit_price - entry_price) * side_mult
    fee_usdt = (qty * entry_price) * FEE_ENTRY + (qty * exit_price) * FEE_EXIT
    net_usdt = pnl_usdt - fee_usdt
    gross_ret_pct = (pnl_usdt / SIZE) * 100.0
    net_ret_pct   = (net_usdt / SIZE) * 100.0

    # 결과 출력
    print("=== RESULT ===")
    print(f"symbol       : {SYMBOL}")
    print(f"side         : {SIDE}")
    print(f"entry_time   : {entry_time}")
    print(f"entry_price  : {entry_price:.2f}")
    print(f"exit_time    : {exit_time}")
    print(f"exit_price   : {exit_price:.2f}")
    print(f"hit          : {hit}")
    print(f"qty          : {qty:.6f}")
    print(f"gross_ret_%  : {gross_ret_pct:.4f}")
    print(f"net_ret_%    : {net_ret_pct:.4f}")
    print(f"fee_entry_%  : {FEE_ENTRY*100:.3f}")
    print(f"fee_exit_%   : {FEE_EXIT*100:.3f}")
    print(f"leverage     : {LEVERAGE}")

    # CSV 저장(옵션) — 청산시각까지만 저장하려면 df_after[df_after["time"] <= exit_time]
    if CSV_PATH:
        path = df_after.copy()
        path["ret_%"] = (path["close"] / entry_price - 1.0) * side_mult * LEVERAGE * 100.0
        path.to_csv(CSV_PATH, index=False, encoding="utf-8")
        print(f"[OK] saved path to {CSV_PATH}")


if __name__ == "__main__":
    main()
