import pandas as pd

# KST 09:00 == UTC 00:00
#ref 는 비교구간, pred는 예측구간 enumerate는 8월 11을 기준으로 후보 블록들을 생성하는 거 
ANCHOR_BASE_UTC = pd.Timestamp("2025-08-11 00:00")  

def pick_blocks(now_ts, step_hours=72):
    step = pd.Timedelta(hours=step_hours)
    n = int((now_ts - ANCHOR_BASE_UTC) // step)
    ref_start = ANCHOR_BASE_UTC + (n - 1) * step 
    ref_end   = ANCHOR_BASE_UTC + n * step
    pred_start = ref_end #
    pred_end   = pred_start + step
    return (ref_start, ref_end), (pred_start, pred_end)

def enumerate_blocks(df, step_hours=72, window_size=18):
    step = pd.Timedelta(hours=step_hours)
    first_ts = df["timestamp"].iloc[0]; last_ts = df["timestamp"].iloc[-1]
    k0 = int((first_ts - ANCHOR_BASE_UTC) // step)
    cur = ANCHOR_BASE_UTC + k0 * step
    if cur > first_ts: cur -= step
    blocks = []
    while cur + step <= last_ts:
        seg = df[(df["timestamp"] >= cur) & (df["timestamp"] < cur + step)]
        if len(seg) >= window_size:
            blocks.append({"start": cur, "end": cur + step})
        cur += step
    return blocks
