"""
alpaca_settings.py — Load .env, expose typed config. .env always wins over system vars.
"""
from __future__ import annotations
import os

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _load_env(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ[key.strip()] = val.strip().strip('"').strip("'")


_load_env(_ENV_PATH)


def _get(k: str, d: str = "") -> str:
    return os.environ.get(k, d)

def _getf(k: str, d: float) -> float:
    try: return float(_get(k, str(d)))
    except ValueError: return d

def _geti(k: str, d: int) -> int:
    try: return int(_get(k, str(d)))
    except ValueError: return d


ALPACA_API_KEY    = _get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = _get("ALPACA_SECRET_KEY")
TRADE_MODE        = _get("TRADE_MODE", "PAPER")   # PAPER | LIVE
UNIVERSE          = [s.strip() for s in _get("UNIVERSE", "SPY,QQQ,AAPL,MSFT,TSLA,NVDA,AMD,AMZN").split(",")]
TRADE_SIZE_USD    = _getf("TRADE_SIZE_USD", 80.0)
MAX_POSITIONS     = _geti("MAX_POSITIONS", 5)
CASH_RESERVE_USD  = _getf("CASH_RESERVE_USD", 100.0)
STOP_LOSS_PCT     = _getf("STOP_LOSS_PCT", 3.0)
TAKE_PROFIT_PCT   = _getf("TAKE_PROFIT_PCT", 6.0)
CYCLE_SEC         = _geti("CYCLE_SEC", 60)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL  = "https://api.alpaca.markets"
BASE_URL  = PAPER_URL if TRADE_MODE == "PAPER" else LIVE_URL
