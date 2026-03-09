"""
alpaca_data.py — Market data via Alpaca API (free tier, IEX feed).

Fetches OHLCV bars and latest quotes for stocks in the universe.
No separate data subscription needed — included with free Alpaca account.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from alpaca_settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

log = logging.getLogger("alpaca_data")


@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Snapshot:
    symbol: str
    price: float
    bars: List[Bar]      # daily bars, oldest → newest
    rsi: float
    ema: float
    atr: float


def _client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def _ema(values: List[float], period: int) -> float:
    if len(values) < period:
        return sum(values) / len(values) if values else 0.0
    k = 2 / (period + 1)
    ema = values[-period]
    for v in values[-period + 1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas[-period:]]
    losses = [-min(d, 0) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def _atr(bars: List[Bar], period: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i].high, bars[i].low, bars[i-1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    trs = trs[-period:]
    return sum(trs) / len(trs) if trs else 0.0


def get_bars(symbol: str, days: int = 60) -> List[Bar]:
    """Fetch daily OHLCV bars for a symbol."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = _client()
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    try:
        req  = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                                start=start, end=end, feed="iex")
        resp = client.get_stock_bars(req)
        raw  = resp.data.get(symbol, [])
    except Exception as exc:
        log.error("get_bars(%s) failed: %s", symbol, exc)
        return []

    bars = []
    for b in raw:
        bars.append(Bar(
            ts=b.timestamp,
            open=float(b.open),
            high=float(b.high),
            low=float(b.low),
            close=float(b.close),
            volume=float(b.volume),
        ))
    bars.sort(key=lambda b: b.ts)
    return bars


def get_latest_price(symbol: str) -> Optional[float]:
    """Fetch latest trade price."""
    from alpaca.data.requests import StockLatestTradeRequest
    client = _client()
    try:
        req  = StockLatestTradeRequest(symbol_or_symbols=symbol, feed="iex")
        resp = client.get_stock_latest_trade(req)
        trade = resp.get(symbol)
        return float(trade.price) if trade else None
    except Exception as exc:
        log.error("get_latest_price(%s) failed: %s", symbol, exc)
        return None


def get_snapshot(symbol: str) -> Optional[Snapshot]:
    """Fetch bars + compute indicators for a symbol."""
    bars = get_bars(symbol, days=60)
    if len(bars) < 20:
        log.warning("%s: only %d bars — skipping", symbol, len(bars))
        return None

    price = get_latest_price(symbol)
    if price is None:
        price = bars[-1].close  # fallback to last bar close

    closes = [b.close for b in bars]
    return Snapshot(
        symbol=symbol,
        price=price,
        bars=bars,
        rsi=_rsi(closes),
        ema=_ema(closes, 20),
        atr=_atr(bars),
    )


def get_all_snapshots(universe: List[str]) -> Dict[str, Snapshot]:
    """Fetch snapshots for all symbols. Skips any that fail."""
    result = {}
    for sym in universe:
        snap = get_snapshot(sym)
        if snap:
            result[sym] = snap
    return result
