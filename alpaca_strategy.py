"""
alpaca_strategy.py — Swing trading signals for stocks. AGGRESSIVE PAPER MODE.

BUY conditions — 3 entry types:
  ENTRY 1 — Dip buy:        RSI < 55 (no EMA required)
  ENTRY 2 — EMA crossover:  price just crossed above EMA + RSI <= 65
  ENTRY 3 — Trend ride:     2 green bars + price above EMA + RSI 45-68

SELL conditions:
  - RSI > 72 + price below EMA (trail exit)
  - Open P&L >= TAKE_PROFIT_PCT
  - Open P&L <= -STOP_LOSS_PCT

HOLD: everything else.
"""
from __future__ import annotations

from dataclasses import dataclass
from alpaca_data import Snapshot


@dataclass
class Signal:
    action: str       # "BUY" | "SELL" | "HOLD"
    symbol: str
    price: float
    rsi: float
    ema: float
    reason: str


def compute_signal(
    snap: Snapshot,
    open_position: bool = False,
    entry_price: float = 0.0,
    stop_loss_pct: float = 3.0,
    take_profit_pct: float = 6.0,
) -> Signal:
    price = snap.price
    rsi   = snap.rsi
    ema   = snap.ema
    sym   = snap.symbol
    bars  = snap.bars

    def sig(action: str, reason: str) -> Signal:
        return Signal(action, sym, price, rsi, ema, reason)

    closes = [b.close for b in bars]
    gap_pct = (price - ema) / ema * 100 if ema > 0 else 0.0

    # ── Exit logic (position open) ──────────────────────────────────
    if open_position and entry_price > 0:
        pnl_pct = (price - entry_price) / entry_price * 100

        if pnl_pct <= -stop_loss_pct:
            return sig("SELL", f"stop_loss {pnl_pct:.1f}%")

        if pnl_pct >= take_profit_pct:
            return sig("SELL", f"take_profit {pnl_pct:.1f}%")

        # Trailing: RSI extreme AND price fell back below EMA
        if rsi > 72 and price < ema:
            return sig("SELL", f"trail_exit rsi={rsi:.1f}")

    # ── Entry logic — AGGRESSIVE ────────────────────────────────────
    if not open_position:

        # ENTRY 1 — Dip buy: RSI below 55 (no EMA required)
        if rsi < 55:
            return sig("BUY", f"oversold rsi={rsi:.1f} gap={gap_pct:.1f}%")

        # ENTRY 2 — EMA crossover: prev bar below EMA, current above, RSI <= 65
        if len(closes) >= 2 and ema > 0:
            prev_below = closes[-2] < ema
            now_above  = price > ema
            if prev_below and now_above and rsi <= 65:
                return sig("BUY", f"ema_cross_up rsi={rsi:.1f} gap={gap_pct:.1f}%")

        # ENTRY 3 — Trend ride: 2 green bars + above EMA + RSI 45-68
        if len(closes) >= 3 and ema > 0:
            two_green = closes[-1] > closes[-2] > closes[-3]
            if two_green and price > ema and 45.0 <= rsi <= 68.0:
                return sig("BUY", f"trend_ride rsi={rsi:.1f} gap={gap_pct:.1f}%")

    return sig("HOLD", "no_signal")
