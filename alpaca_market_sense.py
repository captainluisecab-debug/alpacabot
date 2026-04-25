"""
alpaca_market_sense.py — Stock-market common-sense gates for Alpaca.

Phase A A7. Implements market-specific instincts that don't apply to
Kraken/Solana (24/7 markets):

  - Market-hours gate (regular trading 9:30-16:00 ET, Mon-Fri)
  - Lunch chop block (11:30-13:30 ET) — historically the worst tradeable
    window of the session: low volume, choppy, no edge
  - SPY drift alignment — when SPY is sharply red intraday, don't go
    long components (high correlation drag)
  - Force-flat warning window (15:55-16:00 ET) — alert that positions
    will be auto-flat at close; entries blocked

  - Earnings window avoidance: stub only — no calendar data source wired
    yet. Operator decision: bring in earnings calendar data, or skip.

Wired into alpaca_engine BUY loop via allow_entry(snap, ctx). All gates
are advisory + defaults to permissive when data unavailable.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

log = logging.getLogger("alpaca_market_sense")

# ── Constants ────────────────────────────────────────────────────────

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MIN = 0

LUNCH_START_HOUR = 11
LUNCH_START_MIN = 30
LUNCH_END_HOUR = 13
LUNCH_END_MIN = 30

# Force-flat warning window — last 5 min before close
FLAT_WARN_HOUR = 15
FLAT_WARN_MIN = 55

# SPY drift threshold — if SPY intraday change is below this, block long entries
SPY_DRIFT_BLOCK_PCT = -1.5  # -1.5% SPY intraday → don't long components

# ── ET clock helper ───────────────────────────────────────────────────

def _et_now() -> datetime:
    """Return current time in ET (rough — uses fixed offset, ignores DST switch
    moments). Acceptable for hour-level gating; not used for sub-minute timing."""
    # EDT = UTC-4 most of the year; EST = UTC-5 in winter.
    # We'll use Apr-Oct EDT default and shift in winter via month check.
    now_utc = datetime.now(timezone.utc)
    month = now_utc.month
    # DST roughly Mar 2nd Sun → Nov 1st Sun
    if 3 <= month <= 11:
        offset = timedelta(hours=-4)  # EDT
    else:
        offset = timedelta(hours=-5)  # EST
    return now_utc + offset


def _minute_of_day(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


# ── Individual gates ─────────────────────────────────────────────────

@dataclass
class GateResult:
    allow: bool
    reason: str  # human-readable description ("OK" if allow)


def is_weekday(dt: Optional[datetime] = None) -> bool:
    dt = dt or _et_now()
    return dt.weekday() < 5  # Mon-Fri


def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """True iff inside regular trading session (9:30-16:00 ET, Mon-Fri)."""
    dt = dt or _et_now()
    if not is_weekday(dt):
        return False
    mod = _minute_of_day(dt)
    open_min = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MIN
    close_min = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MIN
    return open_min <= mod < close_min


def is_lunch_chop(dt: Optional[datetime] = None) -> bool:
    """True iff inside the lunch chop window (11:30-13:30 ET)."""
    dt = dt or _et_now()
    if not is_market_hours(dt):
        return False
    mod = _minute_of_day(dt)
    lunch_start = LUNCH_START_HOUR * 60 + LUNCH_START_MIN
    lunch_end = LUNCH_END_HOUR * 60 + LUNCH_END_MIN
    return lunch_start <= mod < lunch_end


def is_force_flat_window(dt: Optional[datetime] = None) -> bool:
    """True iff in the last 5 minutes before close (15:55-16:00 ET).
    Block new entries that won't have time to develop before flat-close.
    """
    dt = dt or _et_now()
    if not is_market_hours(dt):
        return False
    mod = _minute_of_day(dt)
    warn_min = FLAT_WARN_HOUR * 60 + FLAT_WARN_MIN
    close_min = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MIN
    return warn_min <= mod < close_min


def spy_drift_blocking(spy_pct_today: Optional[float]) -> GateResult:
    """If SPY intraday change is below SPY_DRIFT_BLOCK_PCT, block long entries.
    Returns GateResult(allow=False) when SPY is sharply red.
    Fail-open: if spy_pct_today is None, allow (no data → don't block)."""
    if spy_pct_today is None:
        return GateResult(True, "spy_data_unavailable_failopen")
    if spy_pct_today <= SPY_DRIFT_BLOCK_PCT:
        return GateResult(False,
                          f"spy_drift {spy_pct_today:.2f}% <= {SPY_DRIFT_BLOCK_PCT}% — broad market red, skip longs")
    return GateResult(True, f"spy_drift {spy_pct_today:.2f}% OK")


def earnings_window_blocking(symbol: str, earnings_calendar: Optional[dict] = None) -> GateResult:
    """Earnings-window avoidance — stub only.
    earnings_calendar would map symbol -> next_earnings_datetime.
    Without a calendar source, fail-open (allow).

    TODO: wire alpha_vantage or other free earnings calendar feed.
    """
    if not earnings_calendar:
        return GateResult(True, "earnings_calendar_unavailable_failopen")
    next_er = earnings_calendar.get(symbol)
    if not next_er:
        return GateResult(True, f"{symbol}_no_known_earnings")
    try:
        delta_h = (next_er - _et_now()).total_seconds() / 3600
    except Exception:
        return GateResult(True, "earnings_parse_err_failopen")
    if 0 < delta_h < 24:
        return GateResult(False, f"{symbol} earnings in {delta_h:.0f}h — blocking entry")
    return GateResult(True, f"{symbol} earnings_window OK ({delta_h:.0f}h away)")


# ── Composite entry gate ─────────────────────────────────────────────

def allow_entry(symbol: str, *,
                spy_pct_today: Optional[float] = None,
                earnings_calendar: Optional[dict] = None,
                dt: Optional[datetime] = None) -> GateResult:
    """Composite market-sense gate. Returns GateResult(allow, reason).
    Engine BUY loop calls this BEFORE signal-strength + pair_status checks.

    Order of checks (most fundamental first):
      1. Market hours (9:30-16:00 ET, weekday) — outside hours, no entries
      2. Force-flat window (15:55-16:00 ET) — too late to deploy meaningfully
      3. Lunch chop (11:30-13:30 ET) — historically poor tradeable window
      4. SPY drift — broad market red blocks all longs
      5. Earnings window — symbol-specific 24h-before block (stub)
    """
    dt = dt or _et_now()

    if not is_market_hours(dt):
        return GateResult(False, f"market_closed (et={dt.strftime('%H:%M %a')})")

    if is_force_flat_window(dt):
        return GateResult(False, f"force_flat_window (et={dt.strftime('%H:%M')}, <5min to close)")

    if is_lunch_chop(dt):
        return GateResult(False, f"lunch_chop (et={dt.strftime('%H:%M')})")

    g = spy_drift_blocking(spy_pct_today)
    if not g.allow:
        return g

    g = earnings_window_blocking(symbol, earnings_calendar)
    if not g.allow:
        return g

    return GateResult(True, "OK")


def current_posture(dt: Optional[datetime] = None,
                    spy_pct_today: Optional[float] = None) -> dict:
    """Return a dict summarizing what gates are active right now. Used for
    logging + future packet rendering.
    """
    dt = dt or _et_now()
    return {
        "et_time": dt.strftime("%Y-%m-%d %H:%M %a"),
        "is_weekday": is_weekday(dt),
        "is_market_hours": is_market_hours(dt),
        "is_lunch_chop": is_lunch_chop(dt),
        "is_force_flat_window": is_force_flat_window(dt),
        "spy_pct_today": spy_pct_today,
        "spy_blocking_longs": (spy_pct_today is not None
                               and spy_pct_today <= SPY_DRIFT_BLOCK_PCT),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(current_posture(spy_pct_today=-0.5), indent=2))
    print()
    print("allow_entry('AAPL'):", allow_entry("AAPL", spy_pct_today=-0.5).reason)
    print("allow_entry('AAPL') with SPY -2%:", allow_entry("AAPL", spy_pct_today=-2.0).reason)
