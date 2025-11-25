# features.py
import numpy as np
import pandas as pd

#펀딩비 데이터와 기존 4시간 봉 데이터 합치는 함수, 합친 다음에 칼럼 추가하는 함수 z_score구하는 함수 전처리 함수
GLOBAL_Z_COLS = ['close','atr','vol_pct','funding']   # OI 제거
SAFE_FEATS    = ['log_ret','close_z','atr_z','vol_pct_z','funding_z']
FEAT_COLS     = ['close_z','log_ret','atr_z','vol_pct_z','funding_z']
MINMAX_COLS   = ['log_ret','atr_z','vol_pct_z']

def align_external_to_klines(df_kl: pd.DataFrame, df_ext: pd.DataFrame, col: str, tol='12h'):
    base = df_kl[['timestamp']].sort_values('timestamp').reset_index(drop=True)
    if df_ext is None or len(df_ext) == 0:
        out = base.copy(); out[col] = np.nan
    else:
        ext = df_ext[['timestamp', col]].dropna().sort_values('timestamp').reset_index(drop=True)
        out = pd.merge_asof(base, ext, on='timestamp', direction='backward', tolerance=pd.Timedelta(tol))
    out[col] = out[col].ffill(limit=2).fillna(0.0).astype(float)
    return pd.merge(df_kl, out, on='timestamp', how='left')

def add_features(df_kl, df_funding=None):
    df = df_kl.copy()
    df = align_external_to_klines(df, df_funding, 'funding', tol='12h')
    if 'close' not in df.columns:
        if 'close_x' in df.columns: df['close'] = df['close_x']
        elif 'close_y' in df.columns: df['close'] = df['close_y']
        else: raise KeyError("가격 열 'close' 없음")
    df['log_ret'] = np.log(df['close']).diff()
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum((df['high'] - df['close'].shift(1)).abs(),
                               (df['low']  - df['close'].shift(1)).abs()))
    df['atr'] = tr.rolling(14, min_periods=14).mean()
    df['vol_pct'] = df['volume'].pct_change()
    return df

def apply_static_zscore(df: pd.DataFrame, cols, train_end_ts):
    train = df[df["timestamp"] < train_end_ts]
    if train.empty:
        raise ValueError("apply_static_zscore: 학습 구간이 비어 있음")
    mu = train[cols].mean()
    sd = train[cols].std(ddof=0).replace(0, 1e-9)
    z = ((df[cols] - mu) / sd).fillna(0.0)
    z.columns = [c + "_z" for c in cols]
    drop_cols = [c + "_z" for c in cols if (c + "_z") in df.columns]
    return pd.concat([df.drop(columns=drop_cols, errors="ignore"), z], axis=1)

def finalize_preprocessed(df, window_size, warmup_min=20):
    out = df.copy()
    out[SAFE_FEATS] = out[SAFE_FEATS].replace([np.inf, -np.inf], np.nan)
    warmup = max(warmup_min, 14 + 2, window_size)
    if len(out) > warmup:
        out = out.iloc[warmup:].reset_index(drop=True)
    out = out.dropna(subset=SAFE_FEATS).reset_index(drop=True)
    return out

def window_is_finite(df_window):
    return np.isfinite(df_window[SAFE_FEATS].to_numpy()).all()

def window_vector(df_window, L=18):
    X = df_window[FEAT_COLS].to_numpy(dtype=float)  # (L,F)
    for c in MINMAX_COLS:
        j = FEAT_COLS.index(c)
        v = X[:, j]
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        X[:, j] = 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.reshape(-1)
