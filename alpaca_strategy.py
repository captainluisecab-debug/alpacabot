"""
alpaca_strategy.py — Swing trading signals for stocks.

BUY conditions (all must be true):
  - RSI(14) < 40  (oversold dip)
  - Price ≤ EMA(20) * 1.01  (at or below 20-day average)
  - Last bar closed down (dip confirmed)

SELL conditions (any is enough):
  - RSI(14) > 62  (overbought, take profit)
  - Open P&L ≥ TAKE_PROFIT_PCT
  - Open P&L ≤ -STOP_LOSS_PCT

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

    # ── Exit logic (position open) ──────────────────────────────────
    if open_position and entry_price > 0:
        pnl_pct = (price - entry_price) / entry_price * 100

        if pnl_pct <= -stop_loss_pct:
            return sig("SELL", f"stop_loss {pnl_pct:.1f}%")

        if pnl_pct >= take_profit_pct:
            return sig("SELL", f"take_profit {pnl_pct:.1f}%")

        # Trailing: RSI overbought AND price back below EMA → exit
        if rsi > 62 and price < ema:
            return sig("SELL", f"trail_exit rsi={rsi:.1f}")

    # ── Entry logic ─────────────────────────────────────────────────
    if not open_position:
        closes = [b.close for b in bars]
        at_or_below_ema = price <= ema * 1.01
        dipping = len(closes) >= 2 and closes[-1] <= closes[-2]

        if rsi < 40 and at_or_below_ema and dipping:
            gap = (price - ema) / ema * 100
            return sig("BUY", f"oversold rsi={rsi:.1f} ema_gap={gap:.1f}%")

    return sig("HOLD", "no_signal")
