from binance.client import Client
import os
from binance.client import Client
from dotenv import load_dotenv

def connect_binance():
    api_key = "YOUR_API_KEY"       # <- 본인 키
    api_secret = "YOUR_SECRET_KEY" # <- 본인 키
    return Client(api_key, api_secret)

