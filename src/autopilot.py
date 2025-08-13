# src/autopilot.py
# ------------------------------------------------------------
# Fully autonomous paper-trading scalper (Alpaca)
# - EMA(FAST/SLOW) + Bollinger(20, STD) bounce/reject signals
# - ATR-based SL/TP bracket orders
# - Fractional LONGs allowed (via notional); SHORTs are whole-share only
# - Per-trade dollar cap, max concurrent positions, daily/weekly drawdown halts
# - Entry windows and symbol sanitation
# - Minimal "ML" gate stub you can replace later
# ------------------------------------------------------------

import os, time, math, logging
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv
from rl.bandit_selector import ContextualBandit
from strategies.registry import ARMS, StratParams

# ---------- Alpaca (alpaca-py) ----------
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ========== Config / ENV loading ==========
ROOT = Path(__file__).resolve().parents[1]         # project root (C:\TradingAI)
ENV_PATH = ROOT / "config" / ".env.paper"          # config\.env.paper
load_dotenv(dotenv_path=str(ENV_PATH), override=True)

def _get(name, default=None, cast=None):
    v = os.getenv(name, default)
    if cast is None:
        return v
    try:
        return cast(v)
    except Exception:
        return default

def _get_bool(name, default=False):
    v = str(os.getenv(name, str(int(default)))).strip().lower()
    return v in {"1","true","yes","y","on"}

def _parse_symbols(raw):
    parts = [p.strip().upper() for p in str(raw or "").replace(";", ",").split(",")]
    # keep only clean alphanumerics (avoids accidental code strings)
    return [s for s in parts if s and s.isalnum()]

# Broker creds
API_KEY = _get("ALPACA_API_KEY")
API_SEC = _get("ALPACA_API_SECRET")
BASE_URL = _get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not API_SEC:
    raise SystemExit(f"[CONFIG] Missing ALPACA_API_KEY/SECRET in {ENV_PATH}")

# Market / symbols
SYMBOLS = _parse_symbols(_get("SYMBOLS", "SPY,AAPL,MSFT,QQQ"))
TZNAME  = _get("MARKET_TIMEZONE", "America/New_York")
NY      = pytz.timezone(TZNAME)

# Strategy knobs
SCALP_EMA_FAST  = int(_get("SCALP_EMA_FAST", 4))
SCALP_EMA_SLOW  = int(_get("SCALP_EMA_SLOW", 18))
SCALP_BB_WINDOW = int(_get("SCALP_BB_WINDOW", 20))
SCALP_BB_STD    = float(_get("SCALP_BB_STD", 1.5))
ML_MIN_PROB     = float(_get("ML_MIN_PROB", 0.52))
POLL_SECONDS    = int(_get("POLL_SECONDS", 20))

# Risk / sizing
MAX_OPEN            = int(_get("MAX_OPEN_POSITIONS", 3))
POS_DOLLARS_CAP     = float(_get("POSITION_DOLLARS_CAP", 15000))
MAX_RISK_PCT_TRADE  = float(_get("MAX_RISK_PCT_PER_TRADE", 0.006))  # reserved if you add ATR sizing

# Exits (ATR-based)
SL_ATR_MULT = float(_get("STOP_LOSS_ATR_MULT", 1.15))
TP_ATR_MULT = float(_get("TAKE_PROFIT_ATR_MULT", 2.00))

# Session protection
DAILY_MAX_LOSS   = float(_get("DAILY_MAX_LOSS_PCT", 2.0))
WEEKLY_MAX_LOSS  = float(_get("WEEKLY_MAX_LOSS_PCT", 5.0))
HALT_ON_DRAWDOWN = _get_bool("HALT_ON_MAX_DRAWDOWN", True)

# Fractional rules
ALLOW_FRAC_LONGS   = _get_bool("ALLOW_FRACTIONAL_LONGS", True)
DISABLE_FRAC_SHORT = _get_bool("DISABLE_FRACTIONAL_SHORTS", True)
MIN_SHORT_QTY      = int(_get("MIN_SHORT_QTY", 1))

# Entry windows / filters
ENTRY_WINDOWS   = _get("ENTRY_WINDOWS", "0930-1130,1400-1600")
SKIP_FIRST_MIN  = int(_get("SKIP_FIRST_MINUTES", 3))
ATR_PCT_MIN     = float(_get("ATR_PCT_RANGE_MIN", 0.0))
ATR_PCT_MAX     = float(_get("ATR_PCT_RANGE_MAX", 999.0))

# Logs
LOG_DIR = Path(_get("LOG_DIR", str(ROOT / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / "autopilot.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
def log(msg):
    print(msg, flush=True)
    logging.info(msg)

# ========== Alpaca clients ==========
trading = TradingClient(API_KEY, API_SEC, paper=True)
dataapi = StockHistoricalDataClient(API_KEY, API_SEC)

# ========== Time helpers ==========
def now_ny():
    return datetime.now(tz=NY)

def market_open():
    try:
        return bool(trading.get_clock().is_open)
    except Exception:
        return False

def within_entry_window(t: datetime) -> bool:
    s = t.strftime("%H%M")
    for win in ENTRY_WINDOWS.split(","):
        win = win.strip()
        if not win or "-" not in win: 
            continue
        a, b = win.split("-")
        if a <= s <= b:
            return True
    return False

# ========== Data / indicators ==========
def fetch_minute_bars(symbol: str, minutes: int = 400):
    end = now_ny()
    start = end - timedelta(minutes=minutes + 5)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        limit=minutes + 5
    )
    bars = dataapi.get_stock_bars(req).df
    if bars is None or bars.empty:
        return pd.DataFrame()
    if isinstance(bars.index, pd.MultiIndex):
        df = bars.xs(symbol, level=0).copy()
    else:
        df = bars.copy()
    # ensure tz-aware NY
    df.index = df.index.tz_convert(NY) if df.index.tzinfo else df.index.tz_localize(NY)
    df.rename(columns={"open":"o","high":"h","low":"l","close":"c","volume":"v"}, inplace=True)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    # EMAs
    out["ema_fast"] = out["c"].ewm(span=SCALP_EMA_FAST, adjust=False).mean()
    out["ema_slow"] = out["c"].ewm(span=SCALP_EMA_SLOW, adjust=False).mean()
    # Bollinger
    ma = out["c"].rolling(SCALP_BB_WINDOW).mean()
    sd = out["c"].rolling(SCALP_BB_WINDOW).std(ddof=0)
    out["bb_mid"] = ma
    out["bb_up"]  = ma + SCALP_BB_STD * sd
    out["bb_dn"]  = ma - SCALP_BB_STD * sd
    # ATR (simple)
    tr = pd.concat([
        (out["h"] - out["l"]),
        (out["h"] - out["c"].shift()).abs(),
        (out["l"] - out["c"].shift()).abs()
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    return out

# ========== Signal + ML gate ==========
def simple_prob_from_bands(c, bb_up, bb_dn) -> float:
    # Confidence ~ closeness to bands (touches tend to be actionable for mean-revert scalps)
    if pd.isna(bb_up) or pd.isna(bb_dn):
        return 0.0
    mid = (bb_up + bb_dn) / 2.0
    half = (bb_up - bb_dn) / 2.0
    if half <= 1e-6:
        return 0.0
    z = min(1.0, abs(c - mid) / half)  # [0..1]
    return float(0.4 + 0.6 * z)        # 0.4 at mid, ~1.0 at band

def make_signal(prev, row):
    # Long: bounce from/below lower band while short EMA > long EMA
    # Short: reject from/above upper band while short EMA < long EMA
    if any(pd.isna(row[["ema_fast","ema_slow","bb_up","bb_dn","atr","c"]])):
        return None
    if row["atr"] <= 0:
        return None

    long_sig  = (row["c"] > prev["c"]) and (prev["c"] <= prev["bb_dn"]) and (row["ema_fast"] > row["ema_slow"])
    short_sig = (row["c"] < prev["c"]) and (prev["c"] >= prev["bb_up"]) and (row["ema_fast"] < row["ema_slow"])

    if long_sig:
        side = "buy"
    elif short_sig:
        side = "sell"
    else:
        return None

    # ML gate (stub): boost confidence when near bands
    p = simple_prob_from_bands(row["c"], row["bb_up"], row["bb_dn"])
    if p < ML_MIN_PROB:
        return None
    return side, float(row["c"]), float(row["atr"]), p

# ========== Sizing / Orders (no fractional shorts) ==========
def calc_qty(side: str, price: float, dollar_target: float):
    if price <= 0 or dollar_target <= 0:
        return 0, False  # qty_or_notional, use_notional
    raw = dollar_target / price
    if side == "buy":
        if ALLOW_FRAC_LONGS:
            # Notional path → broker computes fractional qty for us.
            # Return a rounded qty only for logs.
            q = float(Decimal(raw).quantize(Decimal("0.01"), rounding=ROUND_DOWN))
            return q, True
        else:
            return int(math.floor(raw)), False
    # Short: whole shares only
    q = int(math.floor(raw))
    if q < MIN_SHORT_QTY:
        return 0, False
    return q, False

def place_order(symbol: str, side: str, price: float, atr: float, equity: float):
    # hard cap; simple equity guard (you can replace with ATR-risk sizing later)
    dollar_target = min(POS_DOLLARS_CAP, equity * 0.20)
    qty_or_notional, use_notional = calc_qty(side, price, dollar_target)
    if qty_or_notional == 0:
        log(f"[SKIP] {symbol} {side} qty=0 after fractional/short guards")
        return None

    # Bracket SL/TP (server-side)
    sl = price - SL_ATR_MULT * atr if side == "buy" else price + SL_ATR_MULT * atr
    tp = price + TP_ATR_MULT * atr if side == "buy" else price - TP_ATR_MULT * atr
    take_profit = TakeProfitRequest(limit_price=round(tp, 2))
    stop_loss   = StopLossRequest(stop_price=round(sl, 2))

    if use_notional:
        # Fractional LONG via notional
        req = MarketOrderRequest(
            symbol=symbol,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            notional=str(round(dollar_target, 2)),
            order_type=OrderType.MARKET,
            take_profit=take_profit,
            stop_loss=stop_loss
        )
    else:
        req = MarketOrderRequest(
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            qty=str(int(qty_or_notional)),
            order_type=OrderType.MARKET,
            take_profit=take_profit,
            stop_loss=stop_loss
        )

    order = trading.submit_order(order_data=req)
    log(f"[ORDER] {symbol} {side} "
        f"{'$'+str(round(dollar_target,2)) if use_notional else str(qty_or_notional)} "
        f"@{price:.2f} SL={sl:.2f} TP={tp:.2f}")
    return order

# ========== Session / PnL guards ==========
def open_positions_count() -> int:
    try:
        return len(trading.get_all_positions())
    except Exception:
        return 0

def equity_snapshot():
    acct = trading.get_account()
    eq = float(acct.equity)
    last_eq = float(acct.last_equity) if acct.last_equity is not None else eq
    day_pl = eq - last_eq
    return eq, day_pl

def drawdown_halted(day_pl_pct: float, week_pl_pct: float) -> bool:
    if not HALT_ON_DRAWDOWN:
        return False
    if (-day_pl_pct) >= DAILY_MAX_LOSS:
        log(f"[HALT] Daily loss {day_pl_pct:.2f}% <= -{DAILY_MAX_LOSS}%")
        return True
    if (-week_pl_pct) >= WEEKLY_MAX_LOSS:
        log(f"[HALT] Weekly loss {week_pl_pct:.2f}% <= -{WEEKLY_MAX_LOSS}%")
        return True
    return False

# ========== Main loop ==========
def main():
    log("==== 🚀 AUTOPILOT START ====")
    log(f"[CONFIG] Loaded env: {ENV_PATH}")
    log(f"[CONFIG] Symbols: {SYMBOLS}")

    # Baselines for day/week drawdown
    try:
        start_day_equity = float(trading.get_account().equity)
    except Exception:
        start_day_equity = 100000.0

    while True:
        try:
            if not market_open():
                time.sleep(POLL_SECONDS)
                continue

            tny = now_ny()
            if not within_entry_window(tny):
                time.sleep(POLL_SECONDS)
                continue

            # Skip opening noise
            if tny.hour == 9 and tny.minute < (30 + SKIP_FIRST_MIN):
                time.sleep(POLL_SECONDS)
                continue

            # Risk rails
            eq, day_pl = equity_snapshot()
            day_pl_pct = (day_pl / start_day_equity) * 100.0 if start_day_equity > 0 else 0.0
            week_pl_pct = day_pl_pct  # simple placeholder; wire weekly baseline if you want
            if drawdown_halted(day_pl_pct, week_pl_pct):
                time.sleep(POLL_SECONDS * 3)
                continue

            if open_positions_count() >= MAX_OPEN:
                time.sleep(POLL_SECONDS)
                continue

            for symbol in SYMBOLS:
                if open_positions_count() >= MAX_OPEN:
                    break

                df = fetch_minute_bars(symbol, minutes=max(400, SCALP_BB_WINDOW * 4))
                if df.empty or len(df) < max(SCALP_BB_WINDOW, SCALP_EMA_SLOW) + 5:
                    continue
                df = compute_indicators(df)

                # Liquidity/volatility guard via ATR%
                last = df.iloc[-1]
                if last["c"] <= 0 or pd.isna(last["atr"]) or last["atr"] <= 0:
                    continue
                atr_pct = (last["atr"] / last["c"]) * 100.0
                if atr_pct < ATR_PCT_MIN or atr_pct > ATR_PCT_MAX:
                    continue

                side_pack = make_signal(df.iloc[-2], last)
                if not side_pack:
                    continue

                side, price, atr, prob = side_pack
                placed = place_order(symbol, side, price, atr, eq)
                if placed:
                    log(f"[SIGNAL] {symbol} {side} prob={prob:.2f} atr%={atr_pct:.2f}")

            time.sleep(POLL_SECONDS)

        except Exception as e:
            log(f"[LOOP_ERR] {e}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
