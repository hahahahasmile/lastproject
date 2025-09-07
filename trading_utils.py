# trading_utils.py
import numpy as np
import pandas as pd

def adjust_pct_by_price_level(current_price, base_price, pct_seq, ratio_min=1.5):
    if base_price is None or current_price is None: 
        return np.array(pct_seq, dtype=float)
    if base_price <= 0 or current_price <= 0:       
        return np.array(pct_seq, dtype=float)
    price_ratio = current_price / base_price
    if price_ratio >= ratio_min:
        return np.array(pct_seq, dtype=float) / price_ratio
    return np.array(pct_seq, dtype=float)

def make_entry_at(df, ts, rule="다음봉 시가"):
    seg = df[df["timestamp"] >= ts]
    if len(seg) == 0: 
        return None, None
    if rule == "현재 로직(진입봉 종가)":
        entry_time = seg["timestamp"].iloc[0]
        entry_price = float(seg["close"].iloc[0])
    elif rule == "다음봉 시가":
        if len(seg) < 2:
            entry_time = seg["timestamp"].iloc[0]
            entry_price = float(seg["open"].iloc[0])
        else:
            entry_time = seg["timestamp"].iloc[1]
            entry_price = float(seg["open"].iloc[1])
    else:  # "OHLC 평균(보수적)"
        entry_time = seg["timestamp"].iloc[0]
        hi = float(seg["high"].iloc[0]); lo = float(seg["low"].iloc[0]); cl = float(seg["close"].iloc[0])
        entry_price = (hi + lo + cl) / 3.0
    return entry_time, entry_price

def _round_up(x, step):
    if step <= 0: return x
    return np.ceil(x / step) * step

def _round_down(x, step):
    if step <= 0: return x
    return np.floor(x / step) * step

def make_sl_tp(entry_price, side, method="ATR", atr=None,
               sl_pct=-0.015, tp_pct=0.03, k_sl=1.0, k_tp=2.0,
               tick_size=0.1):
    if method.upper() == "ATR":
        if atr is None or atr <= 0: 
            return None, None
        if side == "LONG":
            sl = entry_price - k_sl*atr; tp = entry_price + k_tp*atr
        elif side == "SHORT":
            sl = entry_price + k_sl*atr; tp = entry_price - k_tp*atr
        else:
            return None, None
    else:
        if side == "LONG":
            sl = entry_price * (1.0 + sl_pct); tp = entry_price * (1.0 + tp_pct)
        elif side == "SHORT":
            sl = entry_price * (1.0 - sl_pct); tp = entry_price * (1.0 - tp_pct)
        else:
            return None, None
    if side == "LONG":
        sl = _round_up(sl, tick_size); tp = _round_up(tp, tick_size)
    else:
        sl = _round_down(sl, tick_size); tp = _round_down(tp, tick_size)
    return float(sl), float(tp)

def position_size(equity_usdt, risk_pct, entry_price, sl_price,
                  contract_value=1.0, max_leverage=10.0, max_notional=100_000.0,
                  qty_step=0.001):
    if entry_price is None or sl_price is None or entry_price <= 0:
        return 0.0
    risk_amount = float(equity_usdt) * float(risk_pct)
    loss_frac = abs(entry_price - sl_price) / entry_price
    if loss_frac <= 0: 
        return 0.0
    notional_calc = risk_amount / loss_frac
    notional_cap = min(notional_calc, equity_usdt * max_leverage, max_notional)
    contracts = notional_cap / contract_value
    contracts = _round_down(contracts, qty_step)
    return float(max(0.0, contracts))

def simulate_trade(df, start_ts, end_ts, side,
                   entry_time, entry_price, sl, tp,
                   fee_entry=0.0004, fee_exit=0.0005,
                   slip_entry=0.0003, slip_exit=0.0005,
                   exit_on_close=True):
    if side not in ("LONG", "SHORT"): 
        return None, None, None, None
    if side == "LONG":
        entry_fill = entry_price * (1 + slip_entry)
    else:
        entry_fill = entry_price * (1 - slip_entry)

    path = df[(df["timestamp"] >= entry_time) & (df["timestamp"] <= end_ts)].reset_index(drop=True)
    if len(path) == 0: 
        return None, None, None, None

    exit_time = path["timestamp"].iloc[-1]
    exit_fill = float(path["close"].iloc[-1])
    hit = False

    for i in range(len(path)):
        hi = float(path["high"].iloc[i]); lo = float(path["low"].iloc[i]); ts = path["timestamp"].iloc[i]
        if side == "LONG":
            if sl is not None and lo <= sl:
                exit_time = ts; exit_fill = sl * (1 - slip_exit); hit = True; break
            if tp is not None and hi >= tp:
                exit_time = ts; exit_fill = tp * (1 - slip_exit); hit = True; break
        else:
            if sl is not None and hi >= sl:
                exit_time = ts; exit_fill = sl * (1 + slip_exit); hit = True; break
            if tp is not None and lo <= tp:
                exit_time = ts; exit_fill = tp * (1 + slip_exit); hit = True; break

    if not hit and exit_on_close:
        if side == "LONG": exit_fill = exit_fill * (1 - slip_exit)
        else:              exit_fill = exit_fill * (1 + slip_exit)

    if side == "LONG":
        gross_ret = (exit_fill - entry_fill) / entry_fill * 100.0
    else:
        gross_ret = (entry_fill - exit_fill) / entry_fill * 100.0

    fee_total = (fee_entry + fee_exit) * 100.0
    net_ret = gross_ret - fee_total

    return exit_time, float(exit_fill), float(gross_ret), float(net_ret)
