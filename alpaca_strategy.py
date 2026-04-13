"""
alpaca_strategy.py — Swing trading signals for stocks.

BUY conditions — 2 entry types (quality-first):
  ENTRY 1 — Dip buy:        RSI < 30 + price within 3% of EMA (not freefall)
  ENTRY 2 — Trend ride:     2 green bars + above EMA + RSI 45-65 + gap < 3.0%
                            (RSI ceiling raised from 58 to 65 — trending stocks
                             sit at 60-65 most of the time, the tighter window
                             was filtering all valid signals)

EMA crossover removed — 75% loss rate, generates false signals in choppy markets.

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
        high_since_entry = max((b.high for b in (bars or [])[-20:]), default=price)
        peak_pnl_pct = (high_since_entry - entry_price) / entry_price * 100

        if pnl_pct <= -stop_loss_pct:
            return sig("SELL", f"stop_loss {pnl_pct:.1f}%")

        # Breakeven stop: if position reached +1% at any point, don't let it
        # become a loss. Exit at breakeven if price returns to entry.
        if peak_pnl_pct >= 1.0 and pnl_pct <= 0.0:
            return sig("SELL", f"breakeven_stop (peak was +{peak_pnl_pct:.1f}%, now {pnl_pct:.1f}%)")

        if pnl_pct >= take_profit_pct:
            return sig("SELL", f"take_profit {pnl_pct:.1f}%")

        # Trailing: RSI extreme AND price fell back below EMA
        if rsi > 72 and price < ema:
            return sig("SELL", f"trail_exit rsi={rsi:.1f}")

    # ── Entry logic — QUALITY FIRST ────────────────────────────────
    if not open_position:

        # ENTRY 1 — Dip buy: RSI genuinely oversold + price near EMA (not freefall)
        if rsi < 30 and gap_pct > -3.0:
            return sig("BUY", f"oversold rsi={rsi:.1f} gap={gap_pct:.1f}%")

        # ENTRY 2 — Trend ride: 2 green bars + above EMA + RSI 45-65 + gap < 3%
        # Tightened from 68 to 65: above 65 is overbought territory, not signal
        # Loosened from 58 (2026-04-11): trending stocks sit at RSI 60-65
        if len(closes) >= 3 and ema > 0:
            two_green = closes[-1] > closes[-2] > closes[-3]
            if two_green and price > ema and 45.0 <= rsi <= 65.0 and gap_pct < 3.0:
                return sig("BUY", f"trend_ride rsi={rsi:.1f} gap={gap_pct:.1f}%")

    return sig("HOLD", "no_signal")
