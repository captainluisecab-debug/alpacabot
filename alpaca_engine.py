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

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

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


def _read_supervisor_cmd() -> dict:
    cmd_path = r"C:\Projects\supervisor\commands\alpaca_cmd.json"
    defaults = {"mode": "NORMAL", "size_mult": 1.0, "entry_allowed": True}
    try:
        if os.path.exists(cmd_path):
            with open(cmd_path, encoding="utf-8") as f:
                return {**defaults, **json.load(f)}
    except Exception:
        pass
    return defaults


def _get_next_open() -> float:
    """
    Return seconds until next market open.
    Uses Alpaca clock API. Falls back to a rough calculation.
    """
    try:
        from alpaca.trading.client import TradingClient
        from alpaca_settings import ALPACA_SECRET_KEY
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY,
                               paper=(TRADE_MODE == "PAPER"))
        clock = client.get_clock()
        if clock.is_open:
            return 0.0
        next_open = clock.next_open  # datetime
        now       = datetime.now(timezone.utc)
        secs      = (next_open - now).total_seconds()
        return max(0.0, secs)
    except Exception:
        # Rough fallback: sleep until 9:25 AM ET next weekday
        now = datetime.now(timezone.utc)
        et_offset = timedelta(hours=4)   # EDT approximation
        now_et = now - et_offset
        target_et = now_et.replace(hour=9, minute=25, second=0, microsecond=0)
        if target_et <= now_et:
            target_et += timedelta(days=1)
        # Skip weekends
        while target_et.weekday() >= 5:
            target_et += timedelta(days=1)
        return (target_et - now_et).total_seconds()


def _after_hours_monitor() -> None:
    """
    Read-only after-hours overview. Logs extended-hours prices for universe.
    No trades executed. Just visibility into what is moving overnight.
    """
    log.info("[AFTER-HOURS] Monitoring extended hours prices (no trading)")
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca_settings import ALPACA_SECRET_KEY

        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        req    = StockLatestQuoteRequest(symbol_or_symbols=UNIVERSE)
        quotes = client.get_stock_latest_quote(req)

        for sym in UNIVERSE:
            q = quotes.get(sym)
            if q:
                mid = (float(q.ask_price) + float(q.bid_price)) / 2 if q.ask_price and q.bid_price else 0
                log.info("[AFTER-HOURS] %-6s bid=%.2f ask=%.2f mid=%.2f",
                         sym, float(q.bid_price or 0), float(q.ask_price or 0), mid)
    except Exception as exc:
        log.warning("[AFTER-HOURS] Quote fetch failed: %s", exc)


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

    # ── Supervisor command ──────────────────────────────────────────
    cmd       = _read_supervisor_cmd()
    sup_mode  = cmd.get("mode", "NORMAL")
    size_mult = float(cmd.get("size_mult", 1.0))
    entry_ok  = bool(cmd.get("entry_allowed", True))
    if sup_mode == "DEFENSE":
        log.info("[CYCLE %d] Supervisor: DEFENSE — no new entries", cycle)
        entry_ok = False
    elif sup_mode == "SCOUT":
        size_mult = min(size_mult, 0.5)

    # ── Market hours check ──────────────────────────────────────────
    if not _is_market_open():
        log.info("[CYCLE %d] Market closed — waiting", cycle)
        return

    # ── Account info ────────────────────────────────────────────────
    account = get_account()
    if account is None:
        log.error("[CYCLE %d] Could not fetch account — skipping", cycle)
        return

    cash        = float(account.cash)
    equity      = float(account.equity)
    unrealized  = float(getattr(account, "unrealized_pl", 0) or 0)

    # Track peak equity in state for drawdown calculation (high-water mark)
    if not hasattr(st, "peak_equity") or equity > getattr(st, "peak_equity", 0):
        st.peak_equity = equity
    peak    = getattr(st, "peak_equity", equity)
    dd_pct  = (equity - peak) / peak * 100 if peak > 0 else 0.0

    # Dynamic baseline: use peak equity as baseline so pnl% reflects drawdown from ATH
    baseline = peak if peak > 0 else equity
    pnl_pct  = (equity - baseline) / baseline * 100 if baseline > 0 else 0.0

    win_rate = (st.winning_trades / st.total_trades * 100
                if st.total_trades > 0 else 0.0)

    log.info(
        "[CYCLE %d] equity=$%.2f pnl=$%+.2f (%.1f%%) dd=%.2f%% | "
        "cash=$%.2f open=%d | realized=$%+.2f trades=%d win=%.0f%%",
        cycle, equity, equity - baseline, pnl_pct, dd_pct,
        cash, len(st.positions), st.realized_pnl_usd,
        st.total_trades, win_rate,
    )

    # ── Drawdown guard — scale or pause entries based on dd_pct ─────
    entry_size_from_dd: float
    if dd_pct <= -8.0:
        entry_ok = False
        entry_size_from_dd = TRADE_SIZE_USD
        log.warning("[CYCLE %d] DD guard: portfolio dd=%.1f%% — entries paused", cycle, dd_pct)
    elif dd_pct <= -5.0:
        entry_size_from_dd = TRADE_SIZE_USD * 0.5
        log.info("[CYCLE %d] DD guard: portfolio dd=%.1f%% — half size", cycle, dd_pct)
    else:
        entry_size_from_dd = TRADE_SIZE_USD
    trade_size_override = entry_size_from_dd

    # ── Fetch live positions from Alpaca ────────────────────────────
    live_positions = get_positions()

    # Sync: remove local positions that Alpaca no longer shows
    for sym in list(st.positions.keys()):
        if sym not in live_positions:
            log.warning("[SYNC] %s not in Alpaca positions — removing from local state", sym)
            st.positions.pop(sym)

    # ── Log open positions with live P&L ────────────────────────────
    if st.positions:
        for sym, pos in st.positions.items():
            live = live_positions.get(sym)
            if live:
                live_price  = float(getattr(live, "current_price", pos.entry_price) or pos.entry_price)
                pos_pnl_pct = (live_price - pos.entry_price) / pos.entry_price * 100
                pos_pnl_usd = pos_pnl_pct / 100 * pos.usd_invested
                log.info(
                    "  [POS] %-6s entry=$%.2f now=$%.2f pnl=$%+.2f (%+.1f%%)",
                    sym, pos.entry_price, live_price, pos_pnl_usd, pos_pnl_pct,
                )
            else:
                log.info("  [POS] %-6s entry=$%.2f (no live price)", sym, pos.entry_price)

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
                pnl = record_sell(st, sym, snap.price, reason=signal.reason)
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
        if not entry_ok:
            break
        # Stop-loss strike block: skip if symbol is blocked this cycle
        if sym in st.blocked_until and cycle < st.blocked_until[sym]:
            log.info("[CYCLE %d] %s blocked until cycle %d (stop_loss strikes) — skipping",
                     cycle, sym, st.blocked_until[sym])
            continue
        elif sym in st.blocked_until and cycle >= st.blocked_until[sym]:
            # Block has expired — clean up
            st.blocked_until.pop(sym, None)
            st.stop_loss_strikes.pop(sym, None)

        trade_usd = trade_size_override * size_mult
        available_cash = cash - CASH_RESERVE_USD - (open_count * trade_usd)
        if available_cash < trade_usd:
            break

        log.info("[CYCLE %d] BUY %s @ $%.2f | rsi=%.1f reason=%s (size=$%.0f sup=%s)",
                 cycle, sym, snap.price, snap.rsi, signal.reason, trade_usd, sup_mode)
        fill = buy_notional(sym, trade_usd)
        if fill:
            record_buy(st, sym, snap.price, trade_size_override)
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
    after_hours_logged = False

    while True:
        cycle += 1
        try:
            # ── Smart sleep when market is closed ───────────────────
            secs_to_open = _get_next_open()
            if secs_to_open > 0:
                # After-hours monitor — run once per closed session
                if not after_hours_logged:
                    _after_hours_monitor()
                    after_hours_logged = True

                wake_buffer = 300  # wake 5 min before open
                sleep_secs  = max(60, secs_to_open - wake_buffer)
                wake_time   = datetime.now(timezone.utc) + timedelta(seconds=sleep_secs)
                log.info(
                    "Market closed. Next open in %.1fh — sleeping until %s",
                    secs_to_open / 3600,
                    wake_time.strftime("%H:%M UTC"),
                )
                time.sleep(sleep_secs)
                continue

            # Market is open
            after_hours_logged = False
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
