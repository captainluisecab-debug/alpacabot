"""
alpaca_data.py — Market data via Alpaca API (free tier, IEX feed).

Fetches OHLCV bars and latest quotes for stocks in the universe.
No separate data subscription needed — included with free Alpaca account.
"""
from __future__ import annotations

import logging
import socket
import time as _time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

from alpaca_settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

# Fail-fast on socket stalls. Kernel-level TCP SYN timeout on Windows is
# ~21s; a bad AWS ELB backend IP can still stall that long even with this
# set, but this closes the gap for any Python-level socket that respects
# the default.
socket.setdefaulttimeout(15.0)

# Connect/read timeouts injected into every alpaca-py HTTP call via
# _session.request monkey-patch in _client(). See _client() below.
# connect=5s fails fast if an AWS ELB backend IP is unreachable; read=20s
# tolerates slow IEX feed responses during market peak.
_HTTP_TIMEOUT = (5, 20)

log = logging.getLogger("alpaca_data")


def _retry_fetch(fn: Callable, *args, max_attempts: int = 2, **kwargs):
    """
    Retry a network-bound fetch on transient timeout/connection errors.
    Returns fn's result on success. Raises the last exception if all
    attempts exhausted. Non-network errors (auth, validation, etc.) bubble
    up immediately without retry.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            transient = any(tok in msg for tok in
                            ("timed out", "timeout", "connection", "max retries"))
            if not transient:
                raise
            last_exc = exc
            if attempt < max_attempts:
                sleep_s = 0.75 * attempt  # 0.75s, then 1.5s
                log.warning("[RETRY] %s attempt %d/%d transient failure: %s -- retrying in %.2fs",
                            getattr(fn, "__name__", "fetch"), attempt, max_attempts, exc, sleep_s)
                _time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


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


# Module-level client cache. StockHistoricalDataClient reuses an underlying
# requests.Session with HTTPAdapter connection pool, so reusing the same
# instance across calls keeps TCP+TLS sessions warm and eliminates the
# 24 handshakes/cycle (8 symbols x 3 fetches) that was amplifying flake rate.
_cached_client = None


def _client():
    global _cached_client
    if _cached_client is None:
        from alpaca.data.historical import StockHistoricalDataClient
        _cached_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        # Monkey-patch the underlying requests.Session to always inject a
        # timeout. The alpaca-py SDK otherwise passes timeout=None to
        # urllib3, which causes the "connect timeout=None" pattern seen in
        # ISSUE-010 error logs: with no Python-level timeout, the socket
        # falls back to kernel TCP SYN behavior (~21s on Windows).
        sess = getattr(_cached_client, "_session", None)
        if sess is not None and hasattr(sess, "request"):
            _orig_request = sess.request
            def _request_with_timeout(method, url, **kwargs):
                kwargs.setdefault("timeout", _HTTP_TIMEOUT)
                return _orig_request(method, url, **kwargs)
            sess.request = _request_with_timeout
    return _cached_client


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

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=today_start,
        end=now_utc,
        feed="iex",
    )
    try:
        resp = _retry_fetch(client.get_stock_bars, req)
        raw = resp.data.get(symbol, [])
    except Exception as exc:
        log.warning("get_intraday_bars_today(%s) failed after retries: %s", symbol, exc)
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

    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                           start=start, end=end, feed="iex")
    try:
        resp = _retry_fetch(client.get_stock_bars, req)
        raw  = resp.data.get(symbol, [])
    except Exception as exc:
        log.error("get_bars(%s) failed after retries: %s", symbol, exc)
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
    req = StockLatestTradeRequest(symbol_or_symbols=symbol, feed="iex")
    try:
        resp = _retry_fetch(client.get_stock_latest_trade, req)
        trade = resp.get(symbol)
        return float(trade.price) if trade else None
    except Exception as exc:
        log.error("get_latest_price(%s) failed after retries: %s", symbol, exc)
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
