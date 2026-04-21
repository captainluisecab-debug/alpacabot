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

# HITRUN-PARITY: hit-and-run small-profit compounder (matches Kraken Compounder
# config 0.7% / 15min). Captures fast intraday equity gains before mean-reversion
# drags them back toward breakeven_stop. Tuned for small-account compounding.
QUICK_PROFIT_PCT = 0.007
QUICK_PROFIT_MAX_HOLD_SEC = 900


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
    hold_sec: int = 0,
) -> Signal:
    price = snap.price
    rsi   = snap.rsi
    ema   = snap.ema
    sym   = snap.symbol
    bars  = snap.bars
    atr   = getattr(snap, "atr", 0.0)
    vwap  = getattr(snap, "vwap", 0.0)  # 0.0 = unavailable (pre-market, fetch fail); fail-open

    def sig(action: str, reason: str) -> Signal:
        return Signal(action, sym, price, rsi, ema, reason)

    closes = [b.close for b in bars]
    gap_pct = (price - ema) / ema * 100 if ema > 0 else 0.0
    # VWAP distance in ATRs (advisory, logged for observability). 0.0 when VWAP unavailable.
    vwap_dist_atr = (price - vwap) / atr if (vwap > 0 and atr > 0) else 0.0

    # ── Exit logic (position open) ──────────────────────────────────
    if open_position and entry_price > 0:
        pnl_pct = (price - entry_price) / entry_price * 100
        high_since_entry = max((b.high for b in (bars or [])[-20:]), default=price)
        peak_pnl_pct = (high_since_entry - entry_price) / entry_price * 100

        # HITRUN-PARITY: hit-and-run small quick profit within short hold window.
        # Checked BEFORE stop/breakeven so small gains get captured before they
        # have a chance to reverse. Matches Kraken Compounder (0.7% / 15min).
        if hold_sec > 0 and hold_sec <= QUICK_PROFIT_MAX_HOLD_SEC and pnl_pct >= QUICK_PROFIT_PCT * 100:
            return sig("SELL", f"quick_profit_hitrun pnl={pnl_pct:.2f}% hold={hold_sec}s")

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
        # VWAP is not gated here; oversold dips below VWAP are desirable (buying the dip).
        if rsi < 30 and gap_pct > -3.0:
            return sig("BUY", f"oversold rsi={rsi:.1f} gap={gap_pct:.1f}% vwap_dist={vwap_dist_atr:+.2f}atr")

        # ENTRY 2 — Trend ride: 2 green bars + above EMA + RSI 45-65 + gap < 3%
        # VWAP stretch guard (advisory): if session VWAP is available, block entries
        # stretched > 0.5 ATR above VWAP — classic 'chase' trap. Fail-open when vwap=0.0.
        if len(closes) >= 3 and ema > 0:
            two_green = closes[-1] > closes[-2] > closes[-3]
            if two_green and price > ema and 45.0 <= rsi <= 65.0 and gap_pct < 3.0:
                if vwap > 0 and atr > 0 and price > vwap + 0.5 * atr:
                    return sig("HOLD", f"vwap_extended price=${price:.2f} vwap=${vwap:.2f} dist={vwap_dist_atr:+.2f}atr")
                return sig("BUY", f"trend_ride rsi={rsi:.1f} gap={gap_pct:.1f}% vwap_dist={vwap_dist_atr:+.2f}atr")

    return sig("HOLD", "no_signal")
