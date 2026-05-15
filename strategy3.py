import os
import json
import time
import logging
import pytz
from datetime import datetime

import yfinance as yf
import pandas as pd
import requests

# ------------------------- Configuration -------------------------
SYMBOLS = {
    # Indices
    "JP225": "^N225",
    "NDX100": "^NDX",
    "FTSE100": "^FTSE",
    "DJ30": "^DJI",
    # Forex
    "USDJPY": "USDJPY=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCHF": "USDCHF=X",
    # Commodity
    "XAUUSD": "GC=F"      # Gold
}

INTERVAL = "5m"
PERIOD = "7d"           # enough for 200 EMA (200*5min = 16.7h)

# EMA periods
EMA_SHORT = 20
EMA_MEDIUM = 50
EMA_LONG = 200

# MACD parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Stochastic parameters
STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH = 3

# Levels
BUY_STOCH_LEVEL = 30
SELL_STOCH_LEVEL = 70

# Time filter (WAT = UTC+1, no DST)
TIME_START = 6    # 6:00 AM
TIME_END   = 18   # 6:00 PM

# File to store last notification times
SIGNAL_LOG_FILE = "signals_log.json"

# Telegram (set secrets in GitHub Actions)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Data fetch retries
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------- Helper functions -------------------------
def send_telegram(message: str) -> bool:
    """Send a message via Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials missing")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Telegram message sent")
        return True
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False

def load_signal_log() -> dict:
    """Load the log of last signals per symbol."""
    if not os.path.exists(SIGNAL_LOG_FILE):
        return {}
    with open(SIGNAL_LOG_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_signal_log(log: dict):
    """Save the signal log."""
    with open(SIGNAL_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

def fetch_data_with_retries(ticker: str):
    """Download 5m data with retries to improve reliability."""
    for attempt in range(MAX_RETRIES):
        try:
            session = requests.Session()
            session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            df = yf.download(
                ticker,
                period=PERIOD,
                interval=INTERVAL,
                progress=False,
                session=session,
                timeout=30
            )
            if df.empty:
                logger.warning(f"Empty data for {ticker}, attempt {attempt+1}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY)
                continue
            return df
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {ticker}: {e}")
            time.sleep(RETRY_DELAY)
    logger.error(f"Failed to fetch data for {ticker} after {MAX_RETRIES} attempts")
    return None

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all required indicators in one DataFrame."""
    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=EMA_MEDIUM, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=EMA_LONG, adjust=False).mean()

    exp1 = df["Close"].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = df["Close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD_line"] = exp1 - exp2
    df["MACD_signal"] = df["MACD_line"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["MACD_hist"] = df["MACD_line"] - df["MACD_signal"]

    low_min = df["Low"].rolling(window=STOCH_K).min()
    high_max = df["High"].rolling(window=STOCH_K).max()
    df["%K_raw"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
    df["%K"] = df["%K_raw"].rolling(window=STOCH_SMOOTH).mean()
    df["%D"] = df["%K"].rolling(window=STOCH_D).mean()

    return df

def check_buy_condition(row: pd.Series, prev_row: pd.Series) -> bool:
    if not (row["EMA20"] > row["EMA50"] and row["EMA50"] > row["EMA200"]):
        return False
    if row["MACD_line"] <= 0:
        return False
    if prev_row["%K"] <= prev_row["%D"] and row["%K"] > row["%D"] and row["%K"] >= BUY_STOCH_LEVEL:
        return True
    return False

def check_sell_condition(row: pd.Series, prev_row: pd.Series) -> bool:
    if not (row["EMA20"] < row["EMA50"] and row["EMA50"] < row["EMA200"]):
        return False
    if row["MACD_line"] >= 0:
        return False
    if prev_row["%K"] >= prev_row["%D"] and row["%K"] < row["%D"] and row["%K"] <= SELL_STOCH_LEVEL:
        return True
    return False

def is_within_trading_hours() -> bool:
    wat = pytz.timezone("Africa/Lagos")
    now_wat = datetime.now(wat)
    hour = now_wat.hour
    return TIME_START <= hour < TIME_END

def process_symbol(symbol_name: str, ticker: str, signal_log: dict) -> None:
    logger.info(f"Processing {symbol_name} ({ticker})")

    df = fetch_data_with_retries(ticker)
    if df is None or df.empty:
        return

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = compute_indicators(df)
    df.dropna(inplace=True)

    if len(df) < 2:
        logger.warning(f"Not enough data for {symbol_name}")
        return

    latest = df.iloc[-1]
    previous = df.iloc[-2]
    latest_time = latest.name

    now = datetime.now()
    if (now - latest_time).total_seconds() > 600:
        logger.info(f"Latest candle for {symbol_name} is too old ({latest_time}) – skipping")
        return

    is_buy = check_buy_condition(latest, previous)
    is_sell = check_sell_condition(latest, previous)

    signal = None
    if is_buy:
        signal = "BUY"
    elif is_sell:
        signal = "SELL"

    if signal:
        last_signal_key = f"{symbol_name}_last_time"
        last_signal_time = signal_log.get(last_signal_key)

        if last_signal_time and last_signal_time == str(latest_time):
            logger.info(f"Duplicate signal for {symbol_name} at {latest_time} – skipped")
            return

        signal_log[last_signal_key] = str(latest_time)
        save_signal_log(signal_log)

        msg = (f"📊 <b>{symbol_name}</b>\n"
               f"🔔 {signal} SIGNAL\n"
               f"🕒 {latest_time.strftime('%Y-%m-%d %H:%M:%S')} WAT\n"
               f"📈 EMA: {latest['EMA20']:.2f} > {latest['EMA50']:.2f} > {latest['EMA200']:.2f}\n"
               f"📉 MACD: {latest['MACD_line']:.4f}\n"
               f"⚡ Stoch: %K={latest['%K']:.1f}  %D={latest['%D']:.1f}")
        if signal == "BUY":
            msg += "\n✅ Conditions: EMA alignment + MACD>0 + Stoch cross up @30"
        else:
            msg += "\n❌ Conditions: EMA alignment + MACD<0 + Stoch cross down @70"

        send_telegram(msg)
    else:
        logger.info(f"No signal for {symbol_name} at {latest_time}")

def main():
    if not is_within_trading_hours():
        logger.info("Outside trading hours (6:00-18:00 WAT). Exiting.")
        return

    logger.info("Starting Strategy 3 scan (within trading hours)")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram environment variables not set. Will not send messages.")
    else:
        logger.info("Telegram configured")

    signal_log = load_signal_log()
    for name, ticker in SYMBOLS.items():
        process_symbol(name, ticker, signal_log)

    logger.info("Scan finished")

if __name__ == "__main__":
    main()
