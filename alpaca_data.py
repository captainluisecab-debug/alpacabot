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
    bars: List[Bar]      # daily bars, oldest -> newest
    rsi: float
    ema: float
    atr: float
    vwap: float = 0.0    # session VWAP (intraday); 0.0 = unavailable, strategy falls back to VWAP-agnostic


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


def _session_vwap(bars: List[Bar]) -> float:
    """Compute session VWAP from intraday bars (typical price weighted by volume).

    Returns 0.0 if insufficient data — callers treat 0.0 as 'no VWAP available'
    and fall back to VWAP-agnostic logic.
    """
    if not bars:
        return 0.0
    total_pv = 0.0
    total_v = 0.0
    for b in bars:
        typical = (b.high + b.low + b.close) / 3.0
        total_pv += typical * b.volume
        total_v += b.volume
    if total_v <= 0.0:
        return 0.0
    return total_pv / total_v


def get_intraday_bars_today(symbol: str) -> List[Bar]:
    """Fetch today's 5-minute bars for session VWAP calculation.

    Returns empty list on failure or pre-market — caller treats as 'no VWAP available'.
    Uses IEX feed to stay on free tier; IEX provides 15-min delayed intraday which
    is sufficient for VWAP-as-advisory scoring (not HFT).
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = _client()
    now_utc = datetime.now(timezone.utc)
    # US regular session = 13:30-20:00 UTC (09:30-16:00 ET).
    # Start fetch at 13:00 UTC today (30m safety margin) so VWAP is cleanly session-anchored.
    today_start = now_utc.replace(hour=13, minute=0, second=0, microsecond=0)
    if now_utc < today_start:
        # Pre-market (before 13:00 UTC) — no session data yet today
        return []

    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=today_start,
            end=now_utc,
            feed="iex",
        )
        resp = client.get_stock_bars(req)
        raw = resp.data.get(symbol, [])
    except Exception as exc:
        log.warning("get_intraday_bars_today(%s) failed: %s", symbol, exc)
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
    bars.sort(key=lambda x: x.ts)
    return bars


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
    # Session VWAP (intraday). Fails open: empty bars -> vwap=0.0 -> strategy
    # falls back to VWAP-agnostic entry logic.
    intraday_bars = get_intraday_bars_today(symbol)
    vwap = _session_vwap(intraday_bars)
    return Snapshot(
        symbol=symbol,
        price=price,
        bars=bars,
        rsi=_rsi(closes),
        ema=_ema(closes, 20),
        atr=_atr(bars),
        vwap=vwap,
    )


def get_all_snapshots(universe: List[str]) -> Dict[str, Snapshot]:
    """Fetch snapshots for all symbols. Skips any that fail."""
    result = {}
    for sym in universe:
        snap = get_snapshot(sym)
        if snap:
            result[sym] = snap
    return result
