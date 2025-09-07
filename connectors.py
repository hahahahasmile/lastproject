from binance.client import Client
import os
from binance.client import Client
from dotenv import load_dotenv

def connect_binance():
    api_key = "YOUR_API_KEY"       # <- 본인 키
    api_secret = "YOUR_SECRET_KEY" # <- 본인 키
    return Client(api_key, api_secret)

load_dotenv()

def connect_binance():
    """
    무조건 메인넷(실제 계정) 연결
    """
    key = os.getenv("BINANCE_API_KEY", "")
    sec = os.getenv("BINANCE_API_SECRET", "")
    if not key or not sec:
        raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET 누락")
    # testnet 옵션 제거 → 항상 메인넷
    return Client(key, sec, testnet=False)

def connect_binance_trade():
    """
    실거래용 (동일하게 메인넷)
    """
    return connect_binance()

# ----- Futures Helpers -----
def get_futures_balances(client):
    """
    USDT 선물 지갑 잔고/가용 조회
    """
    bals = client.futures_account_balance()
    usdt = next((b for b in bals if b["asset"] == "USDT"), None)
    return {
        "wallet_balance": float(usdt["balance"]) if usdt else 0.0,
        "available_balance": float(usdt.get("withdrawAvailable", usdt["balance"])) if usdt else 0.0,
    }

def get_futures_positions(client, symbol="BTCUSDT"):
    """
    현재 포지션(일방향 모드 가정)
    """
    infos = client.futures_position_information(symbol=symbol)
    out = []
    for p in infos:
        amt = float(p["positionAmt"])
        if abs(amt) < 1e-12:
            continue
        out.append({
            "symbol": p["symbol"],
            "positionAmt": amt,
            "entryPrice": float(p["entryPrice"]),
            "unRealizedPnL": float(p["unRealizedProfit"]),
            "leverage": int(p["leverage"]),
            "marginType": p["marginType"],
        })
    return out

def ensure_leverage_and_margin(client, symbol="BTCUSDT", leverage=10, cross=True):
    """
    레버리지/마진타입 설정 (cross=True면 교차, False면 격리)
    """
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
    """
    심볼의 가격/수량 스텝 조회
    """
    ex = client.futures_exchange_info()
    info = next(s for s in ex["symbols"] if s["symbol"] == symbol)
    tick_size = step_size = None
    for f in info["filters"]:
        if f["filterType"] == "PRICE_FILTER":
            tick_size = float(f["tickSize"])
        if f["filterType"] == "LOT_SIZE":
            step_size = float(f["stepSize"])
    return tick_size, step_size

def get_spot_balances(client):
    """
    현물(Spot) 지갑 잔고 조회
    """
    bals = client.get_account()
    assets = []
    for b in bals["balances"]:
        free = float(b["free"])
        locked = float(b["locked"])
        total = free + locked
        if total > 0:
            assets.append({
                "asset": b["asset"],
                "free": free,
                "locked": locked,
                "total": total
            })
    return assets