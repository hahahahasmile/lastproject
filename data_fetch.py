# data_fetch.py
import time
import pandas as pd
from datetime import timedelta

#2020년 1월 1일부터 4시간봉으로 데이터 수집 및 펀딩비를 수집한 함수
def fetch_futures_4h_klines(client, symbol="BTCUSDT", interval="4h",
                            start_time="2020-01-01", end_time=None):
    all_data, current_time = [], start_time
    while True:
        klines = client.futures_historical_klines(symbol, interval, current_time, limit=1000)
        if not klines:
            break
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume',
            'close_time','quote_asset_volume','number_of_trades',
            'taker_buy_base_volume','taker_buy_quote_volume','ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)
        all_data.append(df[['timestamp','open','high','low','close','volume']])
        last_time = df['timestamp'].iloc[-1]
        if end_time and last_time >= pd.to_datetime(end_time):
            break
        current_time = str(last_time + timedelta(milliseconds=1))
        time.sleep(0.2)
    full = pd.concat(all_data).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    return full

def fetch_funding_rate(client, symbol="BTCUSDT", start_time="2020-01-01"):
    st_ms = int(pd.Timestamp(start_time, tz='UTC').timestamp()*1000)
    rows, start = [], st_ms
    while True:
        try:
            data = client.futures_funding_rate(symbol=symbol, startTime=start, limit=1000)
        except Exception:
            data = []
        if not data:
            break
        rows += data
        if len(data) < 1000:
            break
        start = data[-1]['fundingTime'] + 1
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame(columns=['timestamp','funding'])
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df['funding'] = df['fundingRate'].astype(float)
    return df[['timestamp','funding']]
