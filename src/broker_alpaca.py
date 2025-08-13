# Thin Alpaca wrapper
import os
from alpaca_trade_api import REST

def get_client():
    return REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_API_SECRET'),
        base_url=os.getenv('ALPACA_BASE_URL','https://paper-api.alpaca.markets')
    )
