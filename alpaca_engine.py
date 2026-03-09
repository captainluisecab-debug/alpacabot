"""
alpaca_engine.py — Main trading loop for Alpaca stock swing bot.

- Runs every CYCLE_SEC during market hours only (9:30 AM–4:00 PM ET, Mon–Fri)
- Fetches snapshots for all universe stocks
- Computes signals, executes buys/sells via Alpaca API
- Tracks state locally for entry prices + P&L

Run:
    python alpaca_engine.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone

# ── Logging setup ───────────────────────────────────────────────────
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][ALPACA] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "alpaca_engine.log"),
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("alpaca_engine")

from alpaca_settings import (
    CASH_RESERVE_USD,
    CYCLE_SEC,
    MAX_POSITIONS,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRADE_MODE,
    TRADE_SIZE_USD,
    UNIVERSE,
    ALPACA_API_KEY,
)
from alpaca_data import get_all_snapshots
from alpaca_strategy import compute_signal
from alpaca_broker import buy_notional, get_account, get_positions, sell_all
from alpaca_state import load_state, record_buy, record_sell, save_state


def _is_market_open() -> bool:
    """Check if US stock market is currently open via Alpaca clock endpoint."""
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(ALPACA_API_KEY,
                               __import__("alpaca_settings").ALPACA_SECRET_KEY,
                               paper=(TRADE_MODE == "PAPER"))
        clock = client.get_clock()
        return clock.is_open
    except Exception:
        # Fallback: rough ET time check (Mon–Fri 9:30–16:00)
        now_et = datetime.now(timezone.utc)
        # ET = UTC-5 (EST) or UTC-4 (EDT) — approximate
        hour = (now_et.hour - 4) % 24
        minute = now_et.minute
        weekday = now_et.weekday()  # 0=Mon, 4=Fri
        if weekday >= 5:
            return False
        open_mins  = 9 * 60 + 30
        close_mins = 16 * 60
        now_mins   = hour * 60 + minute
        return open_mins <= now_mins < close_mins


def _run_cycle(st, cycle: int) -> None:
    st.cycle = cycle

    # ── Market hours check ──────────────────────────────────────────
    if not _is_market_open():
        log.info("[CYCLE %d] Market closed — waiting", cycle)
        return

    # ── Account info ────────────────────────────────────────────────
    account = get_account()
    if account is None:
        log.error("[CYCLE %d] Could not fetch account — skipping", cycle)
        return

    cash   = float(account.cash)
    equity = float(account.equity)
    log.info("[CYCLE %d] equity=$%.2f cash=$%.2f positions=%d",
             cycle, equity, cash, len(st.positions))

    # ── Fetch live positions from Alpaca ────────────────────────────
    live_positions = get_positions()

    # Sync: remove local positions that Alpaca no longer shows
    for sym in list(st.positions.keys()):
        if sym not in live_positions:
            log.warning("[SYNC] %s not in Alpaca positions — removing from local state", sym)
            st.positions.pop(sym)

    # ── Fetch market data + compute signals ─────────────────────────
    snapshots = get_all_snapshots(UNIVERSE)
    if not snapshots:
        log.warning("[CYCLE %d] No snapshots available — skipping", cycle)
        return

    # ── SELL loop — check exits first ──────────────────────────────
    for sym, pos in list(st.positions.items()):
        snap = snapshots.get(sym)
        if snap is None:
            continue
        signal = compute_signal(
            snap,
            open_position=True,
            entry_price=pos.entry_price,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT,
        )
        if signal.action == "SELL":
            log.info("[CYCLE %d] SELL %s @ $%.2f | reason=%s", cycle, sym, snap.price, signal.reason)
            fill = sell_all(sym)
            if fill:
                pnl = record_sell(st, sym, snap.price)
                log.info("[CYCLE %d] %s sold | pnl=$%.2f | realized_total=$%.2f",
                         cycle, sym, pnl, st.realized_pnl_usd)

    # ── BUY loop — find new entries ─────────────────────────────────
    open_count = len(st.positions)
    if open_count >= MAX_POSITIONS:
        log.info("[CYCLE %d] Max positions (%d) reached — no new buys", cycle, MAX_POSITIONS)
        return

    available_cash = cash - CASH_RESERVE_USD
    if available_cash < TRADE_SIZE_USD:
        log.info("[CYCLE %d] Insufficient cash ($%.2f) for new position", cycle, available_cash)
        return

    # Score all symbols: rank by RSI ascending (most oversold first)
    candidates = []
    for sym, snap in snapshots.items():
        if sym in st.positions:
            continue  # already holding
        signal = compute_signal(snap, open_position=False,
                                stop_loss_pct=STOP_LOSS_PCT,
                                take_profit_pct=TAKE_PROFIT_PCT)
        if signal.action == "BUY":
            candidates.append((snap.rsi, sym, snap, signal))

    candidates.sort(key=lambda x: x[0])  # most oversold first

    for _, sym, snap, signal in candidates:
        if open_count >= MAX_POSITIONS:
            break
        available_cash = cash - CASH_RESERVE_USD - (open_count * TRADE_SIZE_USD)
        if available_cash < TRADE_SIZE_USD:
            break

        log.info("[CYCLE %d] BUY %s @ $%.2f | rsi=%.1f reason=%s",
                 cycle, sym, snap.price, snap.rsi, signal.reason)
        fill = buy_notional(sym, TRADE_SIZE_USD)
        if fill:
            record_buy(st, sym, snap.price, TRADE_SIZE_USD)
            open_count += 1

    save_state(st)


def main() -> None:
    if not ALPACA_API_KEY or ALPACA_API_KEY == "YOUR_KEY_HERE":
        log.error("[FATAL] ALPACA_API_KEY not set in .env")
        sys.exit(1)

    log.info("=" * 60)
    log.info("ALPACA SWING BOT")
    log.info("Mode: %s | Universe: %s", TRADE_MODE, ", ".join(UNIVERSE))
    log.info("Trade size: $%.0f | Max positions: %d | Reserve: $%.0f",
             TRADE_SIZE_USD, MAX_POSITIONS, CASH_RESERVE_USD)
    log.info("Stop: %.1f%% | TP: %.1f%% | Cycle: %ds",
             STOP_LOSS_PCT, TAKE_PROFIT_PCT, CYCLE_SEC)
    log.info("=" * 60)

    st = load_state()
    cycle = st.cycle

    while True:
        cycle += 1
        try:
            _run_cycle(st, cycle)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.error("[CYCLE %d] Unhandled error: %s", cycle, exc, exc_info=True)
        time.sleep(CYCLE_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Alpaca bot stopped.")
