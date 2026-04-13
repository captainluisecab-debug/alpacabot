"""
alpaca_engine.py — Main trading loop for Alpaca stock swing bot.

- Runs every CYCLE_SEC during market hours only (9:30 AM–4:00 PM ET, Mon–Fri)
- Fetches snapshots for all universe stocks
- Computes signals, executes buys/sells via Alpaca API
- Tracks state locally for entry prices + P&L

Run:
    python alpaca_engine.py
"""
# SLEEVE: Alpaca — Execute and Obey.
# Does NOT invent policy. Obeys Governor commands. Goal: positive PnL.
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
from alpaca_brain import run_brain as brain_run, load_overrides as brain_overrides


def _read_supervisor_cmd() -> dict:
    _default_cmd_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "supervisor", "commands", "alpaca_cmd.json"
    )
    cmd_path = os.environ.get("ALPACA_CMD_PATH", os.path.normpath(_default_cmd_path))
    defaults = {"mode": "NORMAL", "size_mult": 1.0, "entry_allowed": True}
    try:
        if os.path.exists(cmd_path):
            with open(cmd_path, encoding="utf-8") as f:
                return {**defaults, **json.load(f)}
    except Exception:
        pass
    return defaults


# Regime stability tracking for entry filter
# Persisted in alpaca_state.json to survive restarts (prevents gate bypass)
_sup_mode_since: tuple = ("", 0.0)
_SUP_MODE_MIN_STABLE_SEC = 7200
try:
    import json as _json_init
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpaca_state.json")) as _f:
        _saved_sup = _json_init.load(_f).get("sup_mode_since")
    if _saved_sup and isinstance(_saved_sup, list) and len(_saved_sup) == 2:
        _sup_mode_since = (str(_saved_sup[0]), float(_saved_sup[1]))
except Exception:
    pass


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
        # Guard: if inside core market hours, market is open — return immediately
        if now_et.weekday() < 5 and 9 * 60 + 30 <= now_et.hour * 60 + now_et.minute < 16 * 60:
            return 0.0
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

    # ── Brain overrides — load every cycle, run brain every 10 ─────
    overrides    = brain_overrides()
    stop_loss    = overrides.get("STOP_LOSS_PCT",   STOP_LOSS_PCT)
    take_profit  = overrides.get("TAKE_PROFIT_PCT", TAKE_PROFIT_PCT)
    trade_size   = overrides.get("TRADE_SIZE_USD",  TRADE_SIZE_USD)
    max_pos      = int(overrides.get("MAX_POSITIONS", MAX_POSITIONS))

    # ── Supervisor command ──────────────────────────────────────────
    global _sup_mode_since
    cmd       = _read_supervisor_cmd()
    sup_mode  = cmd.get("mode", "NORMAL")
    size_mult = float(cmd.get("size_mult", 1.0))
    entry_ok  = bool(cmd.get("entry_allowed", True))
    force_flatten = bool(cmd.get("force_flatten", False))
    if sup_mode == "DEFENSE":
        log.info("[CYCLE %d] Supervisor: DEFENSE — no new entries", cycle)
        entry_ok = False
    elif sup_mode == "SCOUT":
        size_mult = min(size_mult, 0.5)

    # Governor FORCE_FLATTEN: close all positions immediately
    if force_flatten and st.positions:
        log.warning("[CYCLE %d] GOVERNOR FORCE_FLATTEN: closing %d positions", cycle, len(st.positions))
        snapshots = get_all_snapshots(UNIVERSE)
        for sym in list(st.positions.keys()):
            snap = snapshots.get(sym) if snapshots else None
            fill = sell_all(sym)
            if fill:
                fill_price = float(fill.get("filled_avg_price") or (snap.price if snap else 0))
                pnl = record_sell(st, sym, fill_price, reason="governor_force_flatten")
                log.info("[CYCLE %d] FLATTEN %s: pnl=$%.2f", cycle, sym, pnl)
        save_state(st)
        return

    # Track supervisor mode stability for regime duration filter
    if _sup_mode_since[0] != sup_mode:
        _sup_mode_since = (sup_mode, time.time())
        # Persist so restarts don't bypass the stability gate
        st.sup_mode_since = list(_sup_mode_since)
    _mode_stable_sec = time.time() - _sup_mode_since[1]
    if sup_mode != "NORMAL" or _mode_stable_sec < _SUP_MODE_MIN_STABLE_SEC:
        if entry_ok and sup_mode == "NORMAL":
            log.info("[CYCLE %d] Supervisor NORMAL for %dm < %dm required — entries blocked until stable",
                     cycle, int(_mode_stable_sec // 60), int(_SUP_MODE_MIN_STABLE_SEC // 60))
        entry_ok = False if sup_mode != "NORMAL" else (entry_ok and _mode_stable_sec >= _SUP_MODE_MIN_STABLE_SEC)

    # ── Market hours check ──────────────────────────────────────────
    if not _is_market_open():
        log.info("[CYCLE %d] Market closed — waiting", cycle)
        save_state(st)
        return

    # ── Account info ────────────────────────────────────────────────
    account = get_account()
    if account is None:
        log.error("[CYCLE %d] Could not fetch account — skipping", cycle)
        save_state(st)
        return

    cash        = float(account.cash)
    equity      = float(account.equity)
    unrealized  = float(getattr(account, "unrealized_pl", 0) or 0)

    # Track peak equity in state for drawdown calculation (high-water mark).
    # st.peak_equity is loaded from / saved to the state file so restarts don't reset it.
    if equity > st.peak_equity:
        st.peak_equity = equity
    peak    = st.peak_equity if st.peak_equity > 0 else equity
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
        entry_size_from_dd = trade_size
        log.warning("[CYCLE %d] DD guard: portfolio dd=%.1f%% — entries paused", cycle, dd_pct)
    elif dd_pct <= -5.0:
        entry_size_from_dd = trade_size * 0.5
        log.info("[CYCLE %d] DD guard: portfolio dd=%.1f%% — half size", cycle, dd_pct)
    else:
        entry_size_from_dd = trade_size
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
        save_state(st)
        return

    # ── SELL loop — check exits first ──────────────────────────────
    BREAKEVEN_TRIGGER_PCT = 1.5   # move stop to breakeven when unrealized >= this %
    BREAKEVEN_STOP_PCT    = 0.1   # effective stop after breakeven trigger (small buffer)

    for sym, pos in list(st.positions.items()):
        snap = snapshots.get(sym)
        if snap is None:
            continue

        # Profit-lock: once position reaches +1.5%, arm breakeven permanently until exit
        _pnl_now = (snap.price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0.0
        if _pnl_now >= BREAKEVEN_TRIGGER_PCT and sym not in st.breakeven_armed:
            st.breakeven_armed.add(sym)
            log.info("[CYCLE %d] %s profit-lock ARMED: pnl=%.1f%% — breakeven stop locked until exit",
                     cycle, sym, _pnl_now)
        _eff_stop = BREAKEVEN_STOP_PCT if sym in st.breakeven_armed else stop_loss

        # Adaptive TP: once armed (proven gainer), lower TP to 3% to capture real gains
        _eff_tp = 3.0 if sym in st.breakeven_armed else take_profit

        signal = compute_signal(
            snap,
            open_position=True,
            entry_price=pos.entry_price,
            stop_loss_pct=_eff_stop,
            take_profit_pct=_eff_tp,
        )
        if signal.action == "SELL":
            log.info("[CYCLE %d] SELL %s @ $%.2f | reason=%s", cycle, sym, snap.price, signal.reason)
            fill = sell_all(sym)
            if fill:
                fill_price = float(fill.get("filled_avg_price") or snap.price)
                proceeds = pos.usd_invested  # approximate; record_sell computes true pnl
                pnl = record_sell(st, sym, fill_price, reason=signal.reason)
                log.info("[CYCLE %d] %s sold | pnl=$%.2f | realized_total=$%.2f",
                         cycle, sym, pnl, st.realized_pnl_usd)
                try:
                    import sys as _sys
                    _sup_path = os.environ.get(
                        "SUPERVISOR_DIR",
                        os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "supervisor"))
                    )
                    if _sup_path not in _sys.path:
                        _sys.path.insert(0, _sup_path)
                    from supervisor_execution import log_execution
                    log_execution("alpaca", sym, "SELL", proceeds, fill_price, pnl, signal.reason)
                except Exception:
                    log.warning("[EXEC_LOG] log_execution failed bot=alpaca side=SELL sym=%s", sym, exc_info=True)

    # ── BUY loop — find new entries ─────────────────────────────────
    open_count = len(st.positions)
    if open_count >= max_pos:
        log.info("[CYCLE %d] Max positions (%d) reached — no new buys", cycle, max_pos)
        save_state(st)
        return

    available_cash = cash - CASH_RESERVE_USD
    if available_cash < trade_size:
        log.info("[CYCLE %d] Insufficient cash ($%.2f) for new position", cycle, available_cash)
        save_state(st)
        return

    # Score all symbols: rank by RSI ascending (most oversold first)
    candidates = []
    for sym, snap in snapshots.items():
        if sym in st.positions:
            continue  # already holding
        signal = compute_signal(snap, open_position=False,
                                stop_loss_pct=stop_loss,
                                take_profit_pct=take_profit)
        if signal.action == "BUY":
            candidates.append((snap.rsi, sym, snap, signal))

    candidates.sort(key=lambda x: x[0])  # most oversold first

    for _, sym, snap, signal in candidates:
        if open_count >= max_pos:
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
            fill_price = float(fill.get("filled_avg_price") or snap.price)
            record_buy(st, sym, fill_price, trade_usd)
            open_count += 1
            try:
                import sys as _sys
                if r"C:\Projects\supervisor" not in _sys.path:
                    _sys.path.insert(0, r"C:\Projects\supervisor")
                from supervisor_execution import log_execution
                log_execution("alpaca", sym, "BUY", trade_usd, fill_price, 0.0, signal.reason)
            except Exception:
                log.warning("[EXEC_LOG] log_execution failed bot=alpaca side=BUY sym=%s", sym, exc_info=True)

    # ── Brain — self-tune parameters every 10 cycles ────────────────
    if cycle % 10 == 0:
        positions_str = ", ".join(
            f"{sym}@${pos.entry_price:.2f}" for sym, pos in st.positions.items()
        ) or "none"
        # Expose live equity/peak on state so brain can read them
        st.equity = equity
        # Local-first: skip brain API call when entries are blocked (zero value)
        if entry_ok:
            brain_run(st, cycle, positions_str)
        elif cycle % 100 == 0:
            log.info("[BRAIN] Skipped — entry_allowed=false, brain adjustments have no effect")

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
