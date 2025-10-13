# news_fetch.py  (교체본)
from __future__ import annotations
import io
import time
import urllib.parse as U
from typing import Optional

import pandas as pd
import requests

try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None  # pytrends 미설치 시에도 안전 통과

# =========================
# 공통 유틸
# =========================

def _normalize_query(q: str) -> str:
    """OR 포함인데 괄호가 없으면 자동으로 ( )를 붙인다."""
    qs = q.strip()
    has_or = " OR " in qs.upper()
    has_paren = qs.startswith("(") and qs.endswith(")")
    return f"({qs})" if has_or and not has_paren else qs

def _resample_4h(df: pd.DataFrame, how="mean", cols=None) -> pd.DataFrame:
    """4시간 격자로 리샘플."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp"])
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    if cols is None:
        cols = [c for c in df.columns if c != "timestamp"]
    agg = {c: ("sum" if how == "sum" else "max" if how == "max" else "mean") for c in cols}
    out = df[cols].resample("4h", label="right", closed="right").agg(agg).reset_index()
    return out

# =========================
# GDELT: DOC 2.0 타임라인 (tone / volume)
# =========================

def _fetch_gdelt_doc_mode(mode: str, q: str, timespan: str = "30d",
                          retries: int = 2, cool: int = 60) -> pd.DataFrame:
    """
    GDELT DOC 2.0 타임라인 호출.
      mode: 'timelinetone' | 'timelinevolraw'
      timespan: '1d' | '7d' | '30d' 등
    429/503 시 지수 백오프. 실패 시 빈 DF.
    """
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    q_norm = _normalize_query(q)
    url = f"{base}?query={U.quote(q_norm)}&mode={mode}&format=CSV&timespan={timespan}"
    headers = {"User-Agent": "Mozilla/5.0 (gdelt-prod/1.0)"}

    backoff = cool
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=25)
            # GDELT는 오류 메시지를 CSV 헤더 1열로 반환하는 경우가 있음 → 200이라도 내용 확인 필요
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                # 오류 메시지 CSV(헤더 1개 열) 방지
                if df.shape[1] == 1:
                    # 대표 오류: OR 쿼리 괄호 요구 / NO MATCHES 등
                    return pd.DataFrame(columns=["timestamp", mode])
                # 일반 케이스: Date/Datetime + Value
                ts_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
                val_col = "Value" if "Value" in df.columns else None
                if ts_col is None or val_col is None:
                    return pd.DataFrame(columns=["timestamp", mode])
                out = df[[ts_col, val_col]].copy()
                out = out.rename(columns={ts_col: "timestamp", val_col: mode})
                out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
                out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                return out
            elif r.status_code in (429, 503) and attempt < retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                return pd.DataFrame(columns=["timestamp", mode])
        except Exception:
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            return pd.DataFrame(columns=["timestamp", mode])

    return pd.DataFrame(columns=["timestamp", mode])

def fetch_gdelt_btc(start_ts: str = "2020-01-01", end_ts: Optional[str] = None,
                    q: str = "bitcoin OR BTC OR cryptocurrency OR ETF OR SEC",
                    timespan: str = "30d") -> pd.DataFrame:
    """
    DOC 2.0 타임라인 기반:
      - timelinetone  → news_tone (평균 톤)
      - timelinevolraw→ news_count(기사 수)
    실패 시 빈 DF.
    """
    tone = _fetch_gdelt_doc_mode("timelinetone", q, timespan=timespan)
    vol  = _fetch_gdelt_doc_mode("timelinevolraw", q, timespan=timespan)

    if tone is not None and not tone.empty:
        tone = _resample_4h(tone.rename(columns={"timelinetone": "news_tone"}), how="mean", cols=["news_tone"])
    else:
        tone = pd.DataFrame(columns=["timestamp", "news_tone"])

    if vol is not None and not vol.empty:
        # timelinevolraw는 카운트 → 합산
        vol = _resample_4h(vol.rename(columns={"timelinevolraw": "news_count"}), how="sum", cols=["news_count"])
    else:
        vol = pd.DataFrame(columns=["timestamp", "news_count"])

    if tone.empty and vol.empty:
        return pd.DataFrame(columns=["timestamp", "news_tone", "news_count"])

    df = pd.merge(tone, vol, on="timestamp", how="outer").sort_values("timestamp").reset_index(drop=True)
    if start_ts:
        df = df[df["timestamp"] >= pd.to_datetime(start_ts, utc=True)]
    if end_ts:
        df = df[df["timestamp"] <= pd.to_datetime(end_ts, utc=True)]
    return df[["timestamp", "news_tone", "news_count"]]

# =========================
# Google Trends: 'Bitcoin' 관심도
# =========================

def fetch_pytrends_btc(start_ts: str = "2020-01-01", end_ts: Optional[str] = None) -> pd.DataFrame:
    """
    pytrends 가용 시 조회. 일/주 단위 희소도를 보간 후 4h로 리샘플.
    실패/미설치 시 빈 DF.
    """
    if TrendReq is None:
        return pd.DataFrame(columns=["timestamp", "trends_btc"])
    try:
        pytrends = TrendReq(hl="en-US", tz=0)
        pytrends.build_payload(["Bitcoin"], timeframe="today 5-y")
        df = pytrends.interest_over_time()
        if df is None or df.empty:
            return pd.DataFrame(columns=["timestamp", "trends_btc"])
        df = df.reset_index().rename(columns={"date": "timestamp", "Bitcoin": "trends_btc"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[["timestamp", "trends_btc"]].set_index("timestamp").sort_index()

        # 시간축 보간 → 1h → 4h
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
        df = df.reindex(full_idx)
        df["trends_btc"] = pd.to_numeric(df["trends_btc"], errors="coerce").interpolate(limit_direction="both")
        df = df.resample("4h", label="right", closed="right").mean().reset_index().rename(columns={"index": "timestamp"})

        if start_ts:
            df = df[df["timestamp"] >= pd.to_datetime(start_ts, utc=True)]
        if end_ts:
            df = df[df["timestamp"] <= pd.to_datetime(end_ts, utc=True)]

        return df[["timestamp", "trends_btc"]]
    except Exception:
        return pd.DataFrame(columns=["timestamp", "trends_btc"])

# =========================
# (옵션) CryptoPanic: 헤드라인 수
# =========================

def fetch_cryptopanic_count(start_ts: str = "2020-01-01", end_ts: Optional[str] = None) -> pd.DataFrame:
    """
    CryptoPanic는 API 키 필요. 미설정/실패 시 빈 DF 반환.
    """
    try:
        return pd.DataFrame(columns=["timestamp", "cpanic_cnt"])
    except Exception:
        return pd.DataFrame(columns=["timestamp", "cpanic_cnt"])

# =========================
# 통합 빌더
# =========================

def build_news_frame(start_ts: str = "2020-01-01",
                     end_ts: Optional[str] = None,
                     q: str = "bitcoin OR BTC OR cryptocurrency OR ETF OR SEC",
                     timespan: str = "30d") -> pd.DataFrame:
    """
    GDELT(tone/count) + Trends + (옵션)CryptoPanic을 outer-merge.
    존재하는 컬럼만 안전하게 반환 (None 포함 금지).
    """
    gdelt  = fetch_gdelt_btc(start_ts, end_ts, q=q, timespan=timespan)
    trend  = fetch_pytrends_btc(start_ts, end_ts)
    cpanic = fetch_cryptopanic_count(start_ts, end_ts)

    df = None
    for part in (gdelt, trend, cpanic):
        if part is None or part.empty:
            continue
        df = part if df is None else pd.merge(df, part, on="timestamp", how="outer")

    if df is None or df.empty:
        # 완전 빈 경우에도 인터페이스 유지
        return pd.DataFrame(columns=["timestamp", "news_tone", "news_count", "trends_btc", "cpanic_cnt"])

    # 시계열 정렬
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ✅ 존재하는 컬럼만 선택 (None 절대 넣지 않기)
    keep = ["timestamp"]
    for c in ("news_tone", "news_count", "trends_btc", "cpanic_cnt"):
        if c in df.columns:
            keep.append(c)

    return df[keep]
