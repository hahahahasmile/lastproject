import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _build_equity_from_log(df_log: pd.DataFrame) -> pd.Series:
    if "eq_after" in df_log.columns:
        eq = pd.to_numeric(df_log["eq_after"], errors="coerce")
    elif {"eq_before", "pnl_usd"}.issubset(df_log.columns):
        base = pd.to_numeric(df_log["eq_before"], errors="coerce").fillna(method="ffill")
        pnl  = pd.to_numeric(df_log["pnl_usd"],  errors="coerce").fillna(0).cumsum()
        base0 = float(base.iloc[0] if len(base) else 1000.0)
        eq = base0 + pnl
    else:
        eq = pd.Series([], dtype=float)
    return eq

def _drawdown_series(eq: pd.Series):
    eq = pd.to_numeric(eq, errors="coerce").fillna(method="ffill")
    eq = eq[eq.notna()]
    if len(eq) == 0:
        return pd.Series([], dtype=float), np.nan, None, None
    peak = eq.cummax()
    dd = (eq - peak) / peak
    i_end = dd.idxmin()
    i_start = (eq.loc[:i_end]).idxmax() if i_end is not None else None
    mdd = float(dd.min()) if len(dd) else np.nan
    return dd, mdd, i_start, i_end

def make_ddonly_fig(df_log: pd.DataFrame):
    # X축
    if "exit_time" in df_log.columns:
        x = pd.to_datetime(df_log["exit_time"], errors="coerce")
    elif "t_entry" in df_log.columns:
        x = pd.to_datetime(df_log["t_entry"], errors="coerce")
    else:
        x = pd.RangeIndex(len(df_log))

    eq = _build_equity_from_log(df_log)
    dd, mdd, _, _ = _drawdown_series(eq)

    x2 = x[:len(dd)]
    fig, ax = plt.subplots(figsize=(10, 3))
    # 드로우다운(음수)만 단독 표시
    ax.fill_between(x2, 0, dd.values, alpha=0.4, step="pre")
    ax.plot(x2, dd.values, linewidth=1)

    ax.set_title(f"MDD (최대낙폭: {mdd:.2%})")
    ax.set_ylabel("Drawdown (ratio)")
    ax.set_xlabel("")
    ax.set_ylim(min(dd.min()*1.1, -0.01), 0.0)  # 아래는 음수, 위는 0
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    return fig
