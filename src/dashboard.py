# src/dashboard.py
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from dotenv import load_dotenv

# Alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# -------- Helpers: secrets/env ----------
def env(name, default=None):
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)

# Local dev: load .env if not on Streamlit Cloud
if not st.secrets:
    ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=str(ROOT / "config" / ".env.paper"), override=True)

API = env("ALPACA_API_KEY")
SEC = env("ALPACA_API_SECRET")
BASE = env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
SYMBOLS = [s.strip() for s in env("SYMBOLS", "SPY,AAPL,MSFT,QQQ,NVDA").split(",") if s.strip()]
TZNAME = env("MARKET_TIMEZONE", "America/New_York")

# -------- UI frame ----------
st.set_page_config(page_title="TradingAI", layout="wide")
st.title("TradingAI")
st.caption("Live trades, equity, positions, and intraday charts. (Paper)")

st_autorefresh = st.sidebar.checkbox("Auto-refresh (15s)", value=True, help="Disable to pause updates")

with st.sidebar.expander("Connection status", expanded=False):
    st.write({"has_API_key": bool(API), "has_secret": bool(SEC), "symbols": SYMBOLS, "tz": TZNAME})

if not (API and SEC):
    st.error("Alpaca credentials not found. Set them in **Advanced settings → Secrets** on Streamlit Cloud.")
    st.stop()

# -------- Clients ----------
try:
    trading = TradingClient(API, SEC, paper=True)
    dataapi = StockHistoricalDataClient(API, SEC)
except Exception as e:
    st.error(f"Failed to create Alpaca clients: {e}")
    st.stop()

# -------- Account metrics ----------
try:
    acct = trading.get_account()
    c1, c2, c3 = st.columns(3)
    c1.metric("Equity", f"${float(acct.equity):,.2f}")
    c2.metric("Buying Power", f"${float(acct.buying_power):,.2f}")
    c3.metric("Status", str(acct.status).replace("AccountStatus.", ""))
except Exception as e:
    st.warning(f"Account fetch failed: {e}")

# -------- Positions ----------
st.subheader("Open Positions")
try:
    pos = trading.get_all_positions()
    if pos:
        dfp = pd.DataFrame([{
            "Symbol": p.symbol,
            "Qty": float(p.qty),
            "Avg": float(p.avg_entry_price),
            "Market": float(p.current_price),
            "Unrealized $": float(p.unrealized_pl)
        } for p in pos])
        st.dataframe(dfp, use_container_width=True)
    else:
        st.write("No open positions.")
except Exception as e:
    st.error(f"Positions error: {e}")

# -------- Recent Orders (FIXED: use QueryOrderStatus) ----------
st.subheader("Recent Orders")
try:
    req = GetOrdersRequest(
        status=QueryOrderStatus.ALL,  # or QueryOrderStatus.CLOSED / OPEN
        limit=50,
        nested=True
    )
    ords = trading.get_orders(filter=req)
    dfo = pd.DataFrame([{
        "Time": (str(o.submitted_at)[:19] if o.submitted_at else ""),
        "Symbol": o.symbol,
        "Side": o.side,
        "Qty/Notional": o.qty or o.notional,
        "Type": o.type,
        "Status": o.status
    } for o in ords])
    st.dataframe(dfo, use_container_width=True)
except Exception as e:
    st.error(f"Orders error: {e}")

# -------- Intraday chart (IEX feed) ----------
symbol = st.sidebar.selectbox("Chart Symbol", SYMBOLS, index=0 if SYMBOLS else None)
if symbol:
    try:
        NY = pytz.timezone(TZNAME)
        end = datetime.now(NY)
        start = end - timedelta(hours=7)

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed="iex"  # avoid SIP restrictions
        )
        bars = dataapi.get_stock_bars(req).df
        if bars is not None and not bars.empty:
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(symbol, level=0)
            fig = go.Figure(data=[go.Candlestick(
                x=bars.index,
                open=bars['open'],
                high=bars['high'],
                low=bars['low'],
                close=bars['close']
            )])
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.subheader(f"{symbol} — Intraday (1-min, IEX)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No intraday bars returned yet (market closed or temporary delay).")
    except Exception as e:
        st.error(f"Chart error for {symbol}: {e}")

# -------- Safe auto-refresh ----------
if st_autorefresh:
    time.sleep(15)
    st.rerun()
