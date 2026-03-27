"""Configuration constants for Polymarket Volatility Monitor."""
import os

MARKET_ID = "1394299"
MARKET_SLUG = "us-forces-enter-iran-by-december-31-573-642-385-371-179-425-262"

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

POLL_INTERVAL_SECONDS = 30
BOLLINGER_PERIOD = 20
BOLLINGER_STD_MULTIPLIER = 1.0  # Aggressive 1-sigma

DB_PATH = os.environ.get("DB_PATH", "volatility.db")
PORT = int(os.environ.get("PORT", "8888"))
