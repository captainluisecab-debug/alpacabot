"""
alpaca_eod_posture.py — End-of-day carry vs flatten decision (Step 7).

Stocks-specific gap. Crypto markets never close, so enzobot doesn't need
this. For Alpaca, holding through the close exposes positions to overnight
gap risk (unhedged). Below $25k account, PDT protection blocks day-trade
sells in the final ~5min window — meaning if compute_signal's breakeven_stop
triggers between 15:55-16:00 ET, the sell fails and the position rolls
overnight unintentionally.

This module decides BEFORE the PDT corner: at 15:50-15:55 ET, evaluate
each held position and emit CARRY (hold overnight as swing) or FLATTEN
(close while broker still allows).

Decision matrix (V1):
  FLATTEN  if earnings within 24h (gap risk too high)
  CARRY    if profit-locked AND pnl_pct >= +1.5% AND portfolio dd > -3%
  FLATTEN  otherwise (not profit-locked, or shallow profit, or deep dd)

V1 has no TRIM (partial sell). Operator approves V2 when partial-sell
broker helper is added.

Wired into alpaca_engine before SELL loop. The set of FLATTEN symbols
overrides compute_signal output inside the SELL loop.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple

log = logging.getLogger("alpaca_eod_posture")

# EOD decision window — 5-min window BEFORE force_flat_window (15:55-16:00)
# starts. We need broker access still, and PDT block kicks in toward the
# end of the session for sub-$25k accounts.
EOD_DECISION_START_HOUR = 15
EOD_DECISION_START_MIN = 50
EOD_DECISION_END_HOUR = 15
EOD_DECISION_END_MIN = 55

# Carry thresholds
CARRY_MIN_PNL_PCT = 1.5     # must be at least this profitable to carry
CARRY_MAX_DD_PCT = -3.0     # don't carry if portfolio is in deep dd


def _minute_of_day(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def is_eod_decision_window(dt: datetime) -> bool:
    """True iff inside the EOD-decision window (15:50-15:55 ET).
    dt MUST already be in ET. Caller passes alpaca_market_sense._et_now().
    """
    if dt.weekday() >= 5:
        return False
    mod = _minute_of_day(dt)
    start = EOD_DECISION_START_HOUR * 60 + EOD_DECISION_START_MIN
    end = EOD_DECISION_END_HOUR * 60 + EOD_DECISION_END_MIN
    return start <= mod < end


def decide_eod_posture(
    *,
    symbol: str,
    pnl_pct: float,
    in_breakeven_armed: bool,
    total_dd_pct: float,
    earnings_in_24h: bool = False,
) -> Tuple[str, str]:
    """Return (decision, reason) for one held position.

    decision: "CARRY" | "FLATTEN"
    reason:   human-readable explanation for log

    Args:
      symbol: ticker
      pnl_pct: current unrealized pnl on the position, in percent (e.g. 1.6 = +1.6%)
      in_breakeven_armed: True if profit-lock has been armed (sticky from earlier in day)
      total_dd_pct: portfolio drawdown percent (negative)
      earnings_in_24h: True if symbol has earnings reported within next 24h
    """
    # Earnings dominates — gap risk too high
    if earnings_in_24h:
        return "FLATTEN", f"earnings_within_24h pnl={pnl_pct:+.1f}%"

    # Carry only if all three conditions hold
    if in_breakeven_armed and pnl_pct >= CARRY_MIN_PNL_PCT and total_dd_pct > CARRY_MAX_DD_PCT:
        return "CARRY", (
            f"profit_locked pnl={pnl_pct:+.1f}% >= {CARRY_MIN_PNL_PCT}% "
            f"dd={total_dd_pct:.1f}% > {CARRY_MAX_DD_PCT}% — carry overnight"
        )

    # Default: flatten
    why = []
    if not in_breakeven_armed:
        why.append("not_profit_locked")
    elif pnl_pct < CARRY_MIN_PNL_PCT:
        why.append(f"pnl={pnl_pct:+.1f}% < {CARRY_MIN_PNL_PCT}%")
    if total_dd_pct <= CARRY_MAX_DD_PCT:
        why.append(f"dd={total_dd_pct:.1f}% <= {CARRY_MAX_DD_PCT}%")
    return "FLATTEN", "carry_blocked: " + " ".join(why or ["default"])


if __name__ == "__main__":
    # Self-test
    import json
    cases = [
        # (symbol, pnl_pct, armed, dd_pct, earnings) -> expected
        ("AAPL", 2.0, True, -1.0, False),    # CARRY — profit-locked, good gain, shallow dd
        ("MSFT", 0.5, False, -1.0, False),   # FLATTEN — not armed
        ("TSLA", -0.5, True, -1.0, False),   # FLATTEN — armed but losing
        ("AMZN", 1.7, True, -5.0, False),    # FLATTEN — deep dd blocks carry
        ("NVDA", 3.0, True, -1.0, True),     # FLATTEN — earnings imminent
        ("SPY",  1.6, True, -2.5, False),    # CARRY — borderline pass
    ]
    for sym, pnl, armed, dd, er in cases:
        d, r = decide_eod_posture(
            symbol=sym, pnl_pct=pnl,
            in_breakeven_armed=armed, total_dd_pct=dd,
            earnings_in_24h=er,
        )
        print(f"{sym}: {d}  ({r})")
