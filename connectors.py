from binance.client import Client
import os
from dotenv import load_dotenv

#실제 바이낸스 계정이랑 연동 거래 기능 구현 잔고 확인

load_dotenv()

def connect_binance():
    key = os.getenv("BINANCE_API_KEY", "")
    sec = os.getenv("BINANCE_API_SECRET", "")
    if not key or not sec:
        raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET 누락")
    # testnet 옵션 제거 → 항상 메인넷
    return Client(key, sec, testnet=False)

def connect_binance_trade():
    return connect_binance()

# ----- Futures Helpers -----
def get_futures_balances(client):
    bals = client.futures_account_balance()
    usdt = next((b for b in bals if b["asset"] == "USDT"), None)
    return {
        "wallet_balance": float(usdt["balance"]) if usdt else 0.0,
        "available_balance": float(usdt.get("withdrawAvailable", usdt["balance"])) if usdt else 0.0,
    }

def get_futures_positions(client, symbol: str | None = None):
    """
    - symbol 지정 시 해당 심볼만, 미지정 시 전체
    - 수량 0 포지션 제외
    - marginType이 없으면 isolated(True/False)로 유추
    - 키/타입을 안전하게 변환
    """
    infos = (client.futures_position_information(symbol=symbol)
             if symbol else client.futures_position_information())

    out = []
    for p in infos:
        try:
            amt = float(p.get("positionAmt", 0) or 0)
        except Exception:
            amt = 0.0
        if abs(amt) < 1e-12:
            continue

        # leverage
        lev_raw = p.get("leverage")
        try:
            lev = int(float(lev_raw)) if lev_raw not in (None, "", "0") else None
        except Exception:
            lev = None

        # margin type: marginType 없으면 isolated로 유추
        mtype = p.get("marginType")
        if not mtype:
            iso = p.get("isolated")
            if iso is True:
                mtype = "ISOLATED"
            elif iso is False:
                mtype = "CROSS"
            else:
                mtype = None

        out.append({
            "symbol": p.get("symbol"),
            "positionAmt": amt,
            "entryPrice": float(p.get("entryPrice", 0) or 0),
            "unRealizedProfit": float(p.get("unRealizedProfit", 0) or 0),
            "leverage": lev,                 # ← 보강
            "marginType": mtype,             # ← 보강
            "markPrice": float(p.get("markPrice", 0) or 0),
            "liquidationPrice": float(p.get("liquidationPrice", 0) or 0),
        })
    return out

def ensure_leverage_and_margin(client, symbol="BTCUSDT", leverage=10, cross=True):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=int(leverage))
    except Exception:
        pass
    try:
        mt = "CROSSED" if cross else "ISOLATED"
        client.futures_change_margin_type(symbol=symbol, marginType=mt)
    except Exception:
        pass

def get_symbol_filters(client, symbol="BTCUSDT"):
    ex = client.futures_exchange_info()
    info = next(s for s in ex["symbols"] if s["symbol"] == symbol)
    tick_size = step_size = None
    for f in info["filters"]:
        if f["filterType"] == "PRICE_FILTER":
            tick_size = float(f["tickSize"])
        if f["filterType"] == "LOT_SIZE":
            step_size = float(f["stepSize"])
    return tick_size, step_size

