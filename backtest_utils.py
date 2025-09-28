# backtest_utils.py
import numpy as np
import pandas as pd

#백테스트 결과를 나타내는 함수들 
def build_equity_curve(trades_df: pd.DataFrame, start_equity: float):
    df = trades_df.copy().sort_values("pred_start").reset_index(drop=True)
    if df.empty:
        return [], []
    dates = [df.loc[0, "pred_start"]] + df["pred_end"].tolist()
    eq = [float(start_equity)]
    for _, r in df.iterrows():
        eq_now = eq[-1]
        if r.get("side") in ("LONG","SHORT") and pd.notnull(r.get("net_ret_%")):
            notional = float(r.get("size_notional", 0.0) or 0.0)
            weight = (notional / eq_now) if eq_now > 0 else 0.0
            eq.append(eq_now * (1.0 + (float(r["net_ret_%"])/100.0)*weight))
        else:
            eq.append(eq_now)
    return dates, eq

def calc_metrics(trades_df: pd.DataFrame, equity_series: list[float]):
    df_tr = trades_df[trades_df["side"].isin(["LONG","SHORT"]) & trades_df["net_ret_%"].notna()].copy()
    n_trades = len(df_tr)
    hit_rate = (df_tr["net_ret_%"] > 0).mean() * 100.0 if n_trades > 0 else 0.0
    avg_win  = df_tr.loc[df_tr["net_ret_%"] > 0, "net_ret_%"].mean() if (df_tr["net_ret_%"] > 0).any() else 0.0
    avg_loss = df_tr.loc[df_tr["net_ret_%"] <= 0, "net_ret_%"].mean() if (df_tr["net_ret_%"] <= 0).any() else 0.0

    # === Sharpe 정확히 계산: 일 단위 리샘플 → √365 ===
    dfx = trades_df.copy().sort_values("pred_start").reset_index(drop=True)
    eq = np.array(equity_series, dtype=float)

    if len(eq) > 1 and not dfx.empty:
        dates = [dfx.loc[0, "pred_start"]] + dfx["pred_end"].tolist()
        s = pd.Series(eq, index=pd.to_datetime(dates))
        eq_daily = s.resample("1D").last().ffill()
        ret_daily = eq_daily.pct_change().dropna()
        sd = ret_daily.std(ddof=1)
        sharpe = 0.0 if sd == 0 else (ret_daily.mean() / sd) * np.sqrt(365)
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(eq) if eq.size else np.array([])
    dd = (eq - peak) / (peak + 1e-12) if eq.size else np.array([0.0])
    mdd = float(dd.min()) if dd.size else 0.0
    total_ret = eq[-1] / eq[0] - 1.0 if eq.size >= 2 else 0.0
    mar = (total_ret / abs(mdd)) if mdd < 0 else float("nan")

    return {
        "n_trades": n_trades,
        "hit_rate": float(hit_rate),
        "avg_win": 0.0 if pd.isna(avg_win) else float(avg_win),
        "avg_loss": 0.0 if pd.isna(avg_loss) else float(avg_loss),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "mar": float(mar),
    }