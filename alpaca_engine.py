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

# Color highlighting for stdout (file handler stays plain)
try:
    sys.path.insert(0, r"C:\Projects")
    from log_colors import attach_color_formatter
    attach_color_formatter(logging.getLogger())
except Exception:
    pass

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
from alpaca_state import load_state, record_buy, record_sell, save_state, sample_postexit
from alpaca_brain import run_brain as brain_run, load_overrides as brain_overrides
from alpaca_brain import PARAM_BOUNDS as ALPACA_PARAM_BOUNDS
from alpaca_market_sense import allow_entry as _market_sense_allow_entry


_BASE = os.path.dirname(os.path.abspath(__file__))
PAIR_STATUS_FILE       = os.path.join(_BASE, "alpaca_pair_status.json")
SENTINEL_OVERRIDE_FILE = os.path.join(_BASE, "alpaca_sentinel_override.json")


# ── ACCOUNT-BLOCKED halt (D-034) ────────────────────────────────────────────
# When Alpaca rejects all orders (40310000 / trading_blocked), HALT order
# attempts instead of spamming 1,529 rejections. Deadlock-proof: a time-cooldown
# auto-clears so the bot probes again (~2x/hr) and resumes the instant Alpaca is
# tradable — no latch, no operator re-arm (mirrors zerobot D-031).
_ACCOUNT_BLOCK_COOLDOWN_SEC = 1800  # 30 min

# ── Late-session entry cutoff (NEAR-EOD GUARD, 2026-06-23, D-approved) ───────
# Block OPENING new positions after this ET wall-clock time. ~5 min ahead of
# the 15:50 EOD-decision window (alpaca_eod_posture.is_eod_decision_window) and
# ~10 min ahead of the 15:55 PDT/force-flat corner. Prevents late entries that
# either churn (immediate flatten) or roll overnight unintentionally (can't
# flatten in the final 5 min + PDT block on sub-$25k accounts). Exits and
# EOD-flatten are NOT affected — this gate lives only in the BUY-candidate loop.
# Revert: set to "16:00" (== close) to disable, or comment the guard block.
NO_ENTRY_AFTER_ET = "15:45"


def _acct_flags_blocked(account) -> bool:
    return bool(getattr(account, "trading_blocked", False)
                or getattr(account, "account_blocked", False)
                or getattr(account, "trade_suspended_by_user", False))


def _escalate_account_blocked(account, cycle) -> None:
    try:
        from escalation_client import write_escalation
        write_escalation("alpacabot", {
            "problem_code": "ACCOUNT_BLOCKED", "urgency": "HIGH",
            "context": {"cycle": cycle,
                        "trading_blocked": getattr(account, "trading_blocked", None),
                        "account_blocked": getattr(account, "account_blocked", None),
                        "trade_suspended_by_user": getattr(account, "trade_suspended_by_user", None)},
            "question": ("Alpaca account is rejecting ALL orders (40310000 / trading_blocked). "
                         "alpacabot HALTED order attempts (auto-probes ~every 30m, auto-resumes "
                         "when tradable). Operator: fix API-key trade permission / account "
                         "trade-suspend flag at Alpaca."),
        })
    except Exception as e:  # noqa: BLE001
        log.error("[HALT] account-blocked escalation failed: %s", e)


def _mark_account_blocked(st, account, cycle) -> None:
    """Set/refresh the halt flag + cooldown; escalate ONCE per block episode."""
    now = time.time()
    was = bool(st.meta.get("account_blocked")) and now < float(st.meta.get("account_blocked_until", 0) or 0)
    st.meta["account_blocked"] = True
    st.meta["account_blocked_until"] = now + _ACCOUNT_BLOCK_COOLDOWN_SEC
    if not was:
        log.error("[HALT] ACCOUNT_BLOCKED — suppressing orders ~%dm (cycle %d)",
                  _ACCOUNT_BLOCK_COOLDOWN_SEC // 60, cycle)
        _escalate_account_blocked(account, cycle)


def _no_recent_real_fills(st, max_age_days: int = 2) -> bool:
    """True if the bot HAS traded before but had no real fill (sell) in N days —
    the frozen-data condition that must NOT drive auto-tune (D-034)."""
    last = 0
    for v in (st.meta.get("last_exit_ts") or {}).values():
        try:
            last = max(last, int(v or 0))
        except (TypeError, ValueError):
            continue
    if last == 0:
        return False  # never sold — don't over-gate a fresh bot
    return (time.time() - last) > max_age_days * 86400


def _read_pair_status_blocks() -> set:
    """Read alpaca_pair_status.json. Return set of ticker symbols that are
    blocked (status in COOLDOWN/PROBATION/DISABLED_SOFT with valid TTL).
    Mirrors enzobot's _apply_pair_status; skips the _meta key.
    """
    if not os.path.exists(PAIR_STATUS_FILE):
        return set()
    try:
        with open(PAIR_STATUS_FILE, encoding="utf-8") as f:
            obj = json.load(f) or {}
    except Exception as exc:
        log.warning("[PAIR_STATUS] read failed: %s", exc)
        return set()

    now_dt = datetime.now(timezone.utc)
    blocked = set()
    for sym, entry in obj.items():
        if sym.startswith("_"):
            continue  # skip _meta
        if not isinstance(entry, dict):
            continue
        status = entry.get("status", "ACTIVE")
        # TTL check — expired rows revert to ACTIVE by ignoring
        ttl = entry.get("ttl_ts")
        if ttl:
            try:
                expiry_dt = datetime.fromisoformat(str(ttl).replace("Z", "+00:00"))
                if now_dt > expiry_dt:
                    continue
            except Exception:
                pass
        if status in ("COOLDOWN", "PROBATION", "DISABLED_SOFT"):
            blocked.add(sym)
    return blocked


def _apply_sentinel_override(overrides: dict) -> dict:
    """Read alpaca_sentinel_override.json and merge into overrides dict.
    Sentinel wins when both brain and sentinel set the same param.
    Validates against ALPACA_PARAM_BOUNDS before applying.
    Honors TTL — expired overrides are ignored (brain baseline takes over).
    Returns a NEW merged dict; does not mutate the input.
    """
    merged = dict(overrides or {})
    if not os.path.exists(SENTINEL_OVERRIDE_FILE):
        return merged
    try:
        with open(SENTINEL_OVERRIDE_FILE, encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        log.warning("[SENTINEL] override read failed: %s", exc)
        return merged

    # TTL check
    ttl_expiry = obj.get("ttl_expiry", "")
    try:
        expiry_dt = datetime.fromisoformat(ttl_expiry.replace("Z", "+00:00"))
        if datetime.now(expiry_dt.tzinfo) > expiry_dt:
            return merged
    except Exception:
        return merged

    changes = obj.get("changes", {}) or {}
    applied = {}
    for k, v in changes.items():
        if k not in ALPACA_PARAM_BOUNDS:
            log.warning("[SENTINEL] dropped %s=%s — not in ALPACA_PARAM_BOUNDS", k, v)
            continue
        lo, hi = ALPACA_PARAM_BOUNDS[k]
        try:
            clamped = max(lo, min(hi, float(v)))
            if k == "MAX_POSITIONS":
                clamped = int(clamped)
            merged[k] = clamped
            applied[k] = clamped
        except Exception as exc:
            log.warning("[SENTINEL] %s clamp failed: %s", k, exc)

    if applied:
        log.warning("[SENTINEL] override applied: %s ttl=%s reason=%s",
                    applied, ttl_expiry, obj.get("reason", ""))
    return merged


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
# 30 min stability floor: market hours are 09:30-16:00 ET (6.5h). A 120m
# gate cost up to 2h of a 6.5h session every morning. 30m still filters
# open-bell volatility but leaves 6h of tradable window.
_sup_mode_since: tuple = ("", 0.0)
_SUP_MODE_MIN_STABLE_SEC = 1800
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


_after_hours_sell_armed = set()

def _after_hours_monitor() -> None:
    """
    After-hours overview + risk monitor.
    Logs extended-hours prices. If any HELD position drops >5% from entry,
    arms a market-open sell flag so the first trading cycle exits it.
    """
    global _after_hours_sell_armed
    log.info("[AFTER-HOURS] Monitoring extended hours prices (no trading)")
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca_settings import ALPACA_SECRET_KEY

        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        req    = StockLatestQuoteRequest(symbol_or_symbols=UNIVERSE)
        quotes = client.get_stock_latest_quote(req)

        st = load_state()
        for sym in UNIVERSE:
            q = quotes.get(sym)
            if q:
                mid = (float(q.ask_price) + float(q.bid_price)) / 2 if q.ask_price and q.bid_price else 0
                log.info("[AFTER-HOURS] %-6s bid=%.2f ask=%.2f mid=%.2f",
                         sym, float(q.bid_price or 0), float(q.ask_price or 0), mid)
                if sym in st.positions and mid > 0:
                    entry = st.positions[sym].entry_price
                    if entry > 0:
                        pnl_pct = (mid - entry) / entry * 100
                        if pnl_pct <= -5.0:
                            _after_hours_sell_armed.add(sym)
                            log.warning("[AFTER-HOURS] RISK: %s at %.1f%% from entry ($%.2f -> $%.2f) — "
                                       "SELL ARMED for market open", sym, pnl_pct, entry, mid)
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

    # ── Brain overrides + Sentinel override ─────────────────────────
    # Parity with enzobot: brain baseline, then sentinel tightens (sentinel
    # wins on overlapping keys). Sentinel override is TTL-bounded; expired
    # rows fall back to brain baseline automatically.
    overrides = brain_overrides()
    overrides = _apply_sentinel_override(overrides)
    stop_loss    = overrides.get("STOP_LOSS_PCT",   STOP_LOSS_PCT)
    take_profit  = overrides.get("TAKE_PROFIT_PCT", TAKE_PROFIT_PCT)
    trade_size   = overrides.get("TRADE_SIZE_USD",  TRADE_SIZE_USD)
    max_pos      = int(overrides.get("MAX_POSITIONS", MAX_POSITIONS))
    # A6b wires — defaults are permissive (no behavior change).
    # Sentinel B12-port writes tighter values during loss streaks.
    min_score    = float(overrides.get("MIN_SCORE_TO_TRADE", 50.0))
    time_stop    = int(overrides.get("TIME_STOP_SEC", 0))
    min_hold     = int(overrides.get("MIN_HOLD_SEC",  0))

    # ── Pair status ladder — COOLDOWN/PROBATION/DISABLED_SOFT blocks.
    # Parity with enzobot's pair_status.json. Additional per-ticker
    # protection layered on top of the stop_loss_strikes blocked_until
    # mechanism. TTL-bounded; auto-reverts when TTL expires.
    pair_status_blocked = _read_pair_status_blocks()
    if pair_status_blocked:
        log.info("[CYCLE %d] pair_status blocks: %s", cycle, sorted(pair_status_blocked))

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

    # ── PROOF PIN (P0.1, 2026-07-15, two-key operator-approved) ──────────────
    # WHY: the governor scaled this sleeve's exposure without a key being turned —
    # size_mult drifted 0.24 -> 0.5 -> 0.80 (trade size ~$30 floor -> $68-170, ~5.7x;
    # max deployment ~12% -> ~64% of equity) DURING the proof window, while the trade
    # log was reporting the wrong sign (broker +$2.49 vs log -$0.30). §0 #1: never
    # scale capital before the proof bar clears. This is a CAP, not a freeze:
    #   - the governor can still throttle DOWN below the cap (protection intact)
    #   - DEFENSE can still halt entries entirely (protection intact)
    #   - it can no longer scale UP without a two-key decision
    # Pairs with R-denomination (P1.3), which makes the proof metric size-invariant.
    # REMOVE ONLY by explicit two-key decision. Revert = delete this block (<2 min).
    PROOF_SIZE_MULT_CAP = 0.25
    if size_mult > PROOF_SIZE_MULT_CAP:
        log.info("[CYCLE %d] [PROOF-PIN] size_mult %.2f -> capped %.2f (P0.1)",
                 cycle, size_mult, PROOF_SIZE_MULT_CAP)
    size_mult = min(size_mult, PROOF_SIZE_MULT_CAP)

    # Governor FORCE_FLATTEN: close all positions immediately
    if force_flatten and st.positions:
        log.warning("[CYCLE %d] GOVERNOR FORCE_FLATTEN: closing %d positions", cycle, len(st.positions))
        snapshots = get_all_snapshots(UNIVERSE)
        for sym in list(st.positions.keys()):
            snap = snapshots.get(sym) if snapshots else None
            pos_before = st.positions.get(sym)
            fill = sell_all(sym)
            if isinstance(fill, dict) and fill.get("error") == "account_blocked":
                _mark_account_blocked(st, None, cycle)  # account not yet fetched here
                log.error("[CYCLE %d] [HALT] %s force-flatten denied — ACCOUNT_BLOCKED; halting cycle", cycle, sym)
                save_state(st)
                return
            # PDT-aware guard: tagged dict from broker means PDT block. Skip
            # phantom record_sell. Governor force-flatten gets loud error log.
            if isinstance(fill, dict) and fill.get("error") == "pdt_block":
                log.error("[CYCLE %d] GOVERNOR FORCE_FLATTEN %s BLOCKED by PDT — "
                          "sub-$25k account limit reached; position remains open",
                          cycle, sym)
                continue
            if fill:
                fill_price = float(fill.get("filled_avg_price") or (snap.price if snap else 0))
                proceeds = float(pos_before.usd_invested) if pos_before else 0.0
                pnl = record_sell(st, sym, fill_price, reason="governor_force_flatten")
                log.info("[CYCLE %d] FLATTEN %s: pnl=$%.2f", cycle, sym, pnl)
                try:
                    import sys as _sys
                    _sup_path = os.environ.get(
                        "SUPERVISOR_DIR",
                        os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "supervisor"))
                    )
                    if _sup_path not in _sys.path:
                        _sys.path.insert(0, _sup_path)
                    from supervisor_execution import log_execution
                    log_execution("alpaca", sym, "SELL", proceeds, fill_price, pnl, "governor_force_flatten")
                except Exception:
                    log.warning("[EXEC_LOG] log_execution failed bot=alpaca side=SELL sym=%s reason=governor_force_flatten", sym, exc_info=True)
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

    # ── Account info ────────────────────────────────────────────────
    # Fetched BEFORE market-closed check so canonical state fields stay fresh
    # across overnight / weekend (one API call per closed-period wake is fine).
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

    # Canonical cross-sleeve state fields (ALPACA_STATE_SCHEMA_UNIFY).
    # Set early so ALL save_state() call sites (including early returns for
    # market-closed, no-snapshots, account-fetch-fail) persist current truth.
    st.equity_usd = float(equity)
    st.unrealized_pnl_usd = float(unrealized)
    st.dd_pct = float(dd_pct)

    # ── ACCOUNT-BLOCKED halt (D-034) ────────────────────────────────
    # Proactive: if the account object reports blocked, flag it. Then compute the
    # effective halt (flag set AND within cooldown). If blocked, suppress ALL order
    # logic this cycle (single gate). Auto-resume: when cooldown elapses (and the
    # account no longer reports blocked) the flag clears and we probe again.
    _now_blk = time.time()
    if _acct_flags_blocked(account):
        _mark_account_blocked(st, account, cycle)
    _account_blocked = (bool(st.meta.get("account_blocked"))
                        and _now_blk < float(st.meta.get("account_blocked_until", 0) or 0))
    if st.meta.get("account_blocked") and not _account_blocked:
        st.meta["account_blocked"] = False
        log.warning("[RESUME] account-block cooldown elapsed — probing orders again (cycle %d)", cycle)
    if _account_blocked:
        log.error("[CYCLE %d] [HALT] ACCOUNT_BLOCKED — all order attempts suppressed "
                  "(equity=$%.2f); auto-probe after cooldown", cycle, equity)
        save_state(st)
        return

    # ── Market hours check (AFTER canonical state update) ──────────
    if not _is_market_open():
        log.info("[CYCLE %d] Market closed — waiting (equity=$%.2f dd=%.2f%%)",
                 cycle, equity, dd_pct)
        save_state(st)
        return

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

    # Per-trade size = equity × TARGET_DEPLOY_PCT / max_pos (Kraken-LIVE
    # mirror of risk_one.py:51-101; auto-compounds with equity growth).
    # Operator-approved 2026-05-07 Option B. Replaces fixed-$ trade_size.
    _target_deploy_pct = float(overrides.get("TARGET_DEPLOY_PCT", 0.80))
    _pct_per_slot = equity * _target_deploy_pct / max(1, max_pos)
    # ── Drawdown guard — scale or pause entries based on dd_pct ─────
    entry_size_from_dd: float
    if dd_pct <= -8.0:
        entry_ok = False
        entry_size_from_dd = _pct_per_slot
        log.warning("[CYCLE %d] DD guard: portfolio dd=%.1f%% — entries paused", cycle, dd_pct)
    elif dd_pct <= -5.0:
        entry_size_from_dd = _pct_per_slot * 0.5
        log.info("[CYCLE %d] DD guard: portfolio dd=%.1f%% — half size", cycle, dd_pct)
    else:
        entry_size_from_dd = _pct_per_slot
    trade_size_override = entry_size_from_dd

    # ── Authoritative reconcile vs Alpaca (D-034) ───────────────────
    # get_positions() returns None on API ERROR, {} only when genuinely flat.
    live_positions = get_positions()

    if live_positions is None:
        # API error — do NOT touch local state (empty != error). This is the fix
        # for the phantom-position thrash (nuke-then-readd loop).
        log.warning("[SYNC] get_positions() API error — skipping reconcile (state preserved)")
    else:
        # SAFE per-cycle reconcile (D-034 root-cause fix, 2026-06-09): drop locals the
        # exchange no longer shows (flag if suspiciously young), AND adopt exchange
        # positions not local using Alpaca's REAL avg_entry_price. Adoption NEVER
        # fabricates a price (the fabrication was the cause of the old D-034 thrash);
        # if avg_entry_price is ever missing it flags for manual review, never guesses.
        # The get_positions()->None API-error guard above is preserved (we only get
        # here with a genuine live_positions dict). See alpaca_reconcile.py.
        from alpaca_reconcile import reconcile_positions
        reconcile_positions(st, live_positions, cycle, log)

    # ── Log open positions with live P&L ────────────────────────────
    if st.positions:
        for sym, pos in st.positions.items():
            live = (live_positions or {}).get(sym)  # live_positions may be None on API error
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

    # Post-exit forward-price sampling (OBSERVABILITY ONLY — no trading logic).
    # Piggybacks on snapshots already fetched this cycle; never raises into the
    # trade path. Per _opus_plan_alpaca_postexit_tracking.md.
    try:
        sample_postexit(cycle, lambda _s: (snapshots[_s].price if _s in snapshots else None))
    except Exception as _exc:
        log.warning("[POSTEXIT] sample call failed: %s", _exc)

    # Classify per-symbol regime from current indicators and persist to
    # state so the supervisor can read a stock-market regime that's
    # independent from Kraken's crypto pair_regime. Classification:
    #   TRENDING_UP   : price > EMA20 and RSI > 55
    #   TRENDING_DOWN : price < EMA20 and RSI < 45
    #   RANGING       : otherwise
    _pair_regime = {}
    for _sym, _snap in snapshots.items():
        if _snap.price > _snap.ema and _snap.rsi > 55.0:
            _pair_regime[_sym] = "TRENDING_UP"
        elif _snap.price < _snap.ema and _snap.rsi < 45.0:
            _pair_regime[_sym] = "TRENDING_DOWN"
        else:
            _pair_regime[_sym] = "RANGING"
    st.pair_regime = _pair_regime

    # ── Compute SPY intraday change for market_sense gate (A7) ───────
    _spy_pct_today = None
    try:
        _spy_snap = snapshots.get("SPY")
        if _spy_snap and _spy_snap.bars and len(_spy_snap.bars) >= 2:
            _prev_close = _spy_snap.bars[-2].close
            if _prev_close > 0:
                _spy_pct_today = (_spy_snap.price - _prev_close) / _prev_close * 100
    except Exception:
        _spy_pct_today = None

    # ── After-hours risk sell — execute armed sells from overnight monitor ──
    global _after_hours_sell_armed
    for _armed_sym in list(_after_hours_sell_armed):
        if _armed_sym in st.positions:
            snap = snapshots.get(_armed_sym)
            if snap and snap.price > 0:
                log.warning("[CYCLE %d] AFTER-HOURS SELL ARMED: %s — executing at market open", cycle, _armed_sym)
                fill = sell_all(_armed_sym)
                if isinstance(fill, dict) and fill.get("error") == "account_blocked":
                    _mark_account_blocked(st, account, cycle)
                    log.error("[CYCLE %d] [HALT] %s after-hours sell denied — ACCOUNT_BLOCKED; halting cycle",
                              cycle, _armed_sym)
                    save_state(st)
                    return
                # PDT-aware guard: tagged dict from broker means PDT block.
                # Skip phantom record_sell. After-hours risk exit denied loudly.
                if isinstance(fill, dict) and fill.get("error") == "pdt_block":
                    log.error("[CYCLE %d] AFTER-HOURS SELL %s BLOCKED by PDT — "
                              "armed risk exit denied; remains armed for next open",
                              cycle, _armed_sym)
                    continue
                if fill:
                    fill_price = float(fill.get("filled_avg_price") or snap.price)
                    _ah_pos = st.positions.get(_armed_sym)  # D-045: capture score BEFORE record_sell pops the position
                    _ah_score = float(getattr(_ah_pos, 'entry_score', -1.0)) if _ah_pos else -1.0
                    pnl = record_sell(st, _armed_sym, fill_price, reason="after_hours_risk")
                    log.info("[CYCLE %d] %s after-hours risk sell | pnl=$%.2f", cycle, _armed_sym, pnl)
                    try:
                        from alpaca_outcome_analyzer import log_trade as _log_outcome
                        _log_outcome(symbol=_armed_sym, side="SELL", entry_signal="inherited",
                                     exit_reason="after_hours_risk", pnl_usd=pnl,
                                     entry_price=st.positions.get(_armed_sym, pos).entry_price if _armed_sym in st.positions else 0,
                                     exit_price=fill_price, hold_sec=0, entry_score=_ah_score)
                    except Exception:
                        pass
    _after_hours_sell_armed.clear()

    # ── EOD posture decision (Step 7 — Alpaca→Kraken parity, prevents PDT corner) ──
    # At 15:50-15:55 ET, evaluate held positions BEFORE final 5-min close window.
    # CARRY: profit-locked, decent gain, portfolio not in deep dd → hold overnight.
    # FLATTEN: not profit-locked OR earnings imminent → close while broker still allows.
    # The set of FLATTEN decisions is consumed inside the SELL loop below to
    # synthesize SELL signals that override compute_signal output.
    _eod_flatten_set = set()
    try:
        from alpaca_eod_posture import is_eod_decision_window, decide_eod_posture
        from alpaca_market_sense import _et_now as _ms_et_now
        _et = _ms_et_now()
        if is_eod_decision_window(_et):
            _eod_summary = []
            for _esym, _epos in st.positions.items():
                _esnap = snapshots.get(_esym)
                if not _esnap or not getattr(_esnap, "price", 0):
                    continue
                _epnl_pct = ((_esnap.price - _epos.entry_price) / _epos.entry_price * 100
                             if _epos.entry_price > 0 else 0.0)
                _decision, _ereason = decide_eod_posture(
                    symbol=_esym,
                    pnl_pct=_epnl_pct,
                    in_breakeven_armed=(_esym in st.breakeven_armed),
                    total_dd_pct=dd_pct,
                )
                _eod_summary.append(f"{_esym}={_decision}({_epnl_pct:+.1f}%)")
                if _decision == "FLATTEN":
                    _eod_flatten_set.add(_esym)
                    # PDT awareness: log loud if FLATTEN target is already
                    # PDT-blocked for today — operator should see that the
                    # symbol will roll overnight unintentionally.
                    _pdt_today = (st.meta.get("pdt_blocked_today") or {}).get(_esym)
                    _today_et_str = _et.strftime("%Y-%m-%d")
                    if _pdt_today == _today_et_str:
                        log.info("[CYCLE %d] [EOD] %s PDT-BLOCKED holding overnight "
                                 "(breaks symmetry, expected for sub-$25k accounts)",
                                 cycle, _esym)
            if _eod_summary:
                log.info("[CYCLE %d] [EOD] decisions: %s", cycle, " ".join(_eod_summary))
    except Exception as _eod_ex:
        log.warning("[CYCLE %d] [EOD] posture eval failed: %s", cycle, _eod_ex)

    # ── PDT-block date-rollover clear ──────────────────────────────
    # st.meta["pdt_blocked_today"] = {sym: "YYYY-MM-DD"}. If the stored ET
    # date != today's ET date, drop the entry — next RTH session is fresh.
    # Single revertable line: comment out the assignment to disable the gate.
    try:
        from alpaca_market_sense import _et_now as _ms_et_now2
        _today_et = _ms_et_now2().strftime("%Y-%m-%d")
        _pdt_map = st.meta.get("pdt_blocked_today") or {}
        _stale = [s for s, d in _pdt_map.items() if d != _today_et]
        for _s in _stale:
            _pdt_map.pop(_s, None)
            log.info("[CYCLE %d] [PDT] %s block cleared (date rollover)", cycle, _s)
        if _pdt_map:
            st.meta["pdt_blocked_today"] = _pdt_map
        elif "pdt_blocked_today" in st.meta:
            st.meta["pdt_blocked_today"] = {}
    except Exception as _pdt_ex:
        log.warning("[CYCLE %d] [PDT] rollover-clear failed: %s", cycle, _pdt_ex)

    # ── SELL loop — check exits first ──────────────────────────────
    BREAKEVEN_TRIGGER_PCT = 1.5   # move stop to breakeven when unrealized >= this %
    BREAKEVEN_STOP_PCT    = 0.1   # effective stop after breakeven trigger (small buffer)
    REENTRY_COOLDOWN_SEC  = 300   # PREMONDAY 2026-05-17: per-symbol cooldown
                                   # after non-stop_loss exit. Kills AMD-style 63s
                                   # chains. 5 min gives real market separation
                                   # vs 60s boundary.

    for sym, pos in list(st.positions.items()):
        snap = snapshots.get(sym)
        if snap is None:
            continue

        # Profit-lock: once position reaches +1.5%, arm breakeven permanently until exit
        _pnl_now = (snap.price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0.0
        # F-1 v3 (2026-05-20): track actual peak since entry. Replaces
        # alpaca_strategy.py bars[-20:] computation that included daily-bar
        # highs from BEFORE entry. Updated BEFORE trail evaluation (which
        # happens inside compute_signal a few lines below).
        if _pnl_now > pos.peak_pnl_pct:
            pos.peak_pnl_pct = _pnl_now
        if _pnl_now >= BREAKEVEN_TRIGGER_PCT and sym not in st.breakeven_armed:
            st.breakeven_armed.add(sym)
            log.info("[CYCLE %d] %s profit-lock ARMED: pnl=%.1f%% — breakeven stop locked until exit",
                     cycle, sym, _pnl_now)
        # F-2 (2026-06-28): breakeven 0.1% floor RETIRED. The trail-from-peak in
        # compute_signal (alpaca_strategy.py:111-123) already governs armed winners
        # and always sits above breakeven, so this floor was dead weight whose only
        # possible effect was to re-introduce the +8.6%->$0 strangle. Configured
        # stop_loss is now the sole hard floor for armed positions; the trail does
        # the profit-lock. breakeven_armed is retained (EOD-posture carry uses it).
        _eff_stop = stop_loss

        # TP stays at configured level — let winners run.
        # Breakeven stop already protects capital; lowering TP kills runners.
        _eff_tp = take_profit

        _hold_sec_alp = (int(time.time()) - pos.entry_ts) if (pos and pos.entry_ts) else 0
        signal = compute_signal(
            snap,
            open_position=True,
            entry_price=pos.entry_price,
            stop_loss_pct=_eff_stop,
            take_profit_pct=_eff_tp,
            hold_sec=_hold_sec_alp,
            time_stop_sec=time_stop,   # A6b wire
            min_hold_sec=min_hold,     # A6b wire
            peak_pnl_pct_actual=pos.peak_pnl_pct,  # F-1 v3: real peak since entry
        )
        # Step 7 (Alpaca→Kraken parity): EOD posture override.
        # If pre-SELL decision was FLATTEN, force SELL regardless of compute_signal.
        # Reason: at 15:50-15:55 ET we still have broker access; waiting for
        # compute_signal's breakeven_stop trigger may push the sell into the
        # final 5-min window where PDT protection blocks the order.
        if sym in _eod_flatten_set and signal.action != "SELL":
            from types import SimpleNamespace
            signal = SimpleNamespace(action="SELL", reason="eod_posture_flatten")
        if signal.action == "SELL":
            # ── PDT-block gate ──────────────────────────────────────────
            # If sym was PDT-blocked earlier today, skip the broker call to
            # avoid log noise + wasted API hits. SAFETY: always attempt the
            # sell if reason contains "stop_loss" — operator wants to know
            # if even a real risk exit is blocked. Substring match (lower)
            # mirrors record_sell's own convention (alpaca_state.py).
            try:
                from alpaca_market_sense import _et_now as _ms_et_now3
                _today_et_sl = _ms_et_now3().strftime("%Y-%m-%d")
            except Exception:
                _today_et_sl = ""
            _pdt_map_sl = st.meta.get("pdt_blocked_today") or {}
            _is_pdt_blocked = _pdt_map_sl.get(sym) == _today_et_sl
            _is_stop_loss = "stop_loss" in (signal.reason or "").lower()
            # PREMONDAY 2026-05-17 fix: once-per-session — stop_loss carve-out
            # removed. 653 wasted sell-attempts pre-fix (NVDA 315, AMD 216).
            # PDT block is binding for rest of session regardless of exit reason.
            if _is_pdt_blocked:
                if _is_stop_loss:
                    log.error("[CYCLE %d] [PDT] %s STOP_LOSS sell skipped — "
                              "broker PDT-blocked earlier today; position rolling.",
                              cycle, sym)
                else:
                    log.info("[CYCLE %d] [PDT] %s sell skipped (blocked until next session)",
                             cycle, sym)
                continue
            log.info("[CYCLE %d] SELL %s @ $%.2f | reason=%s", cycle, sym, snap.price, signal.reason)
            fill = sell_all(sym)
            if isinstance(fill, dict) and fill.get("error") == "account_blocked":
                _mark_account_blocked(st, account, cycle)
                log.error("[CYCLE %d] [HALT] %s sell denied — ACCOUNT_BLOCKED; halting cycle", cycle, sym)
                save_state(st)
                return
            # PDT detection on the response — broker returns
            # {"error": "pdt_block", ...} for 40310100. Mark the symbol so
            # subsequent cycles skip the broker call (gate above). For
            # stop_loss reasons, log loud — operator wants to know.
            if isinstance(fill, dict) and fill.get("error") == "pdt_block":
                _pdt_map_sl = st.meta.setdefault("pdt_blocked_today", {})
                _pdt_map_sl[sym] = _today_et_sl
                if _is_stop_loss:
                    log.error("[CYCLE %d] [PDT] %s STOP_LOSS sell BLOCKED — "
                              "real risk exit denied by Alpaca PDT (sub-$25k). "
                              "Position will roll until next session OR price recovers.",
                              cycle, sym)
                else:
                    log.warning("[CYCLE %d] [PDT] %s sell denied (40310100); "
                                "marked blocked_today=%s — will skip until next RTH",
                                cycle, sym, _today_et_sl)
                save_state(st)
                continue
            if fill:
                fill_price = float(fill.get("filled_avg_price") or snap.price)
                proceeds = pos.usd_invested  # approximate; record_sell computes true pnl
                pnl = record_sell(st, sym, fill_price, reason=signal.reason)
                log.info("[CYCLE %d] %s sold | pnl=$%.2f | realized_total=$%.2f",
                         cycle, sym, pnl, st.realized_pnl_usd)
                # PREMONDAY 2026-05-17: stamp last_exit_ts for non-stop_loss
                # exits. BUY loop enforces REENTRY_COOLDOWN_SEC. Stop_loss
                # exits use separate strike-block ladder (st.blocked_until).
                if "stop_loss" not in (signal.reason or "").lower():
                    st.meta.setdefault("last_exit_ts", {})[sym] = int(time.time())
                try:
                    from alpaca_outcome_analyzer import log_trade as _log_outcome
                    _hold = int(time.time()) - pos.entry_ts if pos.entry_ts > 0 else 0
                    _log_outcome(
                        symbol=sym, side="SELL",
                        entry_signal=getattr(pos, '_entry_signal', 'unknown'),
                        exit_reason=signal.reason, pnl_usd=pnl,
                        entry_price=pos.entry_price, exit_price=fill_price,
                        hold_sec=_hold, rsi_at_exit=snap.rsi if snap else 0,
                        entry_score=getattr(pos, 'entry_score', -1.0),  # D-045
                    )
                except Exception:
                    log.warning("[OUTCOME] trade log failed sym=%s", sym, exc_info=True)
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

    # TARGET_DEPLOY_PCT cap (ALPACA_PARAM_BOUNDS_EXPAND wire).
    # Autonomy loop / sentinel can tighten this (e.g., B12-port → 0.25 after
    # a loss streak). Skips BUY loop if current deployed exposure already
    # meets/exceeds the cap. Default 0.80 = permissive; no change from prior
    # behavior unless sentinel writes an override.
    _target_deploy_pct = float(overrides.get("TARGET_DEPLOY_PCT", 0.80))
    _invested = sum(p.usd_invested for p in st.positions.values())
    _deploy_cap_usd = equity * _target_deploy_pct
    if _invested >= _deploy_cap_usd:
        log.info("[CYCLE %d] Deploy cap reached — invested=$%.2f >= cap=$%.2f (%.0f%% of eq $%.2f) — no new buys",
                 cycle, _invested, _deploy_cap_usd, _target_deploy_pct * 100, equity)
        save_state(st)
        return

    # Brain wiring config — feature flag read fresh each cycle.
    # Default ON when file missing. Kill switch: write
    # {"USE_ALPACA_BRAIN": false} to alpaca_brain_wiring.json.
    _wiring_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "alpaca_brain_wiring.json")
    _use_brain = True
    try:
        if os.path.exists(_wiring_path):
            with open(_wiring_path, encoding="utf-8") as _wf:
                _use_brain = bool(json.load(_wf).get("USE_ALPACA_BRAIN", True))
    except Exception:
        _use_brain = True

    available_cash = cash - CASH_RESERVE_USD
    if available_cash < trade_size_override:
        log.info("[CYCLE %d] Insufficient cash ($%.2f) for new position", cycle, available_cash)
        save_state(st)
        return

    # Load outcome analyzer adjustments (symbol blocks + quality data)
    _analyzer_blocks = set()
    try:
        _adj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpaca_score_adjustments.json")
        if os.path.exists(_adj_path):
            _adj = json.load(open(_adj_path))
            _analyzer_blocks = set(_adj.get("recommended_blocks", []))
            if _analyzer_blocks:
                log.info("[ANALYZER] Blocked symbols: %s", _analyzer_blocks)
    except Exception:
        pass

    # ── Trader pre-pass (BRAIN-LEAD) — Step 1 of Alpaca→Kraken parity (2026-04-30) ──
    # Mirrors enzobot/engine.py:1300-1316 trader-primary pattern. When _use_brain=True,
    # run alpaca_path_classifier + alpaca_trader.decide_entry for every snap-eligible
    # symbol BEFORE legacy compute_signal. Trader-BUY symbols become primary candidates;
    # trader-SKIP symbols veto legacy BUY.
    trader_candidates = {}   # sym -> {"snap","classifier","detail"}
    trader_skip_seen = {}    # sym -> reason (suppresses legacy BUY for same sym)
    _legacy_signals = {}     # sym -> Signal (Step 1.5: cache so candidate-builder reuses)
    if _use_brain:
        try:
            from alpaca_data import get_intraday_bars, bars_to_classifier_dicts
            from alpaca_path_classifier import classify_path
            from alpaca_trader import decide_entry as _trader_decide
        except Exception as _imp_ex:
            log.warning("[CYCLE %d] [BRAIN] pre-pass import error — legacy only: %s",
                        cycle, _imp_ex)
        else:
            for _sym, _snap in snapshots.items():
                if _sym in st.positions or _sym in _analyzer_blocks:
                    continue
                try:
                    # Step 1.5 (Alpaca→Kraken parity): pre-compute legacy score
                    # and attach to snap so alpaca_trader.decide_entry's
                    # score_boost formula gets proper input. Without this,
                    # snap.score=0 → score_boost=0.55x → trades hit MIN_TRADE_USD floor.
                    try:
                        _legacy_sig = compute_signal(_snap, open_position=False,
                                                     stop_loss_pct=stop_loss,
                                                     take_profit_pct=take_profit)
                        _legacy_signals[_sym] = _legacy_sig
                        try:
                            _snap.score = float(getattr(_legacy_sig, "score", 0.0))
                        except Exception:
                            pass
                    except Exception:
                        pass
                    _b5  = get_intraday_bars(_sym, 5,  60)
                    _b15 = get_intraday_bars(_sym, 15, 50)
                    _b1h = get_intraday_bars(_sym, 60, 100)
                    _cls = classify_path(
                        _sym,
                        bars_to_classifier_dicts(_b5),
                        bars_to_classifier_dicts(_b15),
                        bars_to_classifier_dicts(_b1h),
                        prior_state=None,
                    )
                    _classifier_result = {
                        "state": _cls.state,
                        "confidence": float(_cls.confidence),
                        "reasons": list(_cls.reasons),
                    }
                    _t_action, _t_detail = _trader_decide(
                        _sym, _snap, _classifier_result, None,
                        st.positions, max_pos,
                    )
                    if _t_action == "BUY" and isinstance(_t_detail, dict):
                        trader_candidates[_sym] = {
                            "snap": _snap,
                            "classifier": _classifier_result,
                            "detail": _t_detail,
                        }
                        log.info("[CYCLE %d] [BRAIN] %s BUY trader-primary | state=%s conf=%.2f size_mult=%.2f",
                                 cycle, _sym, _cls.state, _cls.confidence,
                                 float(_t_detail.get("size_mult", 1.0)))
                    else:
                        trader_skip_seen[_sym] = str(_t_detail)
                        log.info("[CYCLE %d] [BRAIN] %s SKIP (state=%s conf=%.2f): %s",
                                 cycle, _sym, _cls.state, _cls.confidence, _t_detail)
                except Exception as _bex:
                    log.warning("[CYCLE %d] [BRAIN] %s pre-pass error — legacy fallback: %s",
                                cycle, _sym, _bex)

    # ── Build candidate list — trader-primary first, legacy fallback ──
    # Sort key: trader-primary uses (-conf - 100) so highest-conf trader entries
    # always sort ahead of any legacy RSI value (RSI range 0-100, so subtracting
    # 100 guarantees trader keys are negative and rank first).
    candidates = []
    for sym, snap in snapshots.items():
        if sym in st.positions:
            continue
        if sym in _analyzer_blocks:
            log.info("[ANALYZER] %s blocked by outcome analyzer (poor WR)", sym)
            continue

        # Trader-primary path: trader proposed BUY → synthesize signal, sort by conf
        if sym in trader_candidates:
            from types import SimpleNamespace
            _tc = trader_candidates[sym]
            _t_cls = _tc["classifier"]
            _t_detail = _tc["detail"]
            _trader_signal = SimpleNamespace(
                action="BUY",
                reason="trader_primary_buy: " + _t_cls["state"]
                       + "@" + ("%.2f" % _t_cls["confidence"]),
                score=0.0,
                _trader_primary=True,
                _trader_detail=_t_detail,
                _trader_classifier=_t_cls,
            )
            _sort_key = -float(_t_cls["confidence"]) - 100.0
            candidates.append((_sort_key, sym, snap, _trader_signal))
            continue

        # Legacy path: compute_signal (preserved as fallback) — reuse pre-pass cached signal if available
        signal = _legacy_signals.get(sym) if _legacy_signals.get(sym) is not None \
                 else compute_signal(snap, open_position=False,
                                     stop_loss_pct=stop_loss,
                                     take_profit_pct=take_profit)
        if signal.action == "BUY":
            # Trader saw this symbol but said SKIP → suppress legacy BUY
            if _use_brain and sym in trader_skip_seen:
                log.info("[CYCLE %d] %s legacy BUY suppressed by trader veto: %s",
                         cycle, sym, trader_skip_seen[sym])
                continue
            candidates.append((float(snap.rsi), sym, snap, signal))

    candidates.sort(key=lambda x: x[0])  # trader-primary (-conf-100) first, then legacy (RSI ASC)

    for _, sym, snap, signal in candidates:
        if open_count >= max_pos:
            break
        if not entry_ok:
            break
        # NEAR-EOD GUARD (2026-06-23): block NEW entries after NO_ENTRY_AFTER_ET (ET).
        # Reuses alpaca_market_sense._et_now() — same clock as is_eod_decision_window.
        # Affects ONLY opening positions; exits + EOD flatten run elsewhere.
        try:
            from alpaca_market_sense import _et_now as _ms_et_now_entry
            _et_entry = _ms_et_now_entry()
            _cut_h, _cut_m = (int(x) for x in NO_ENTRY_AFTER_ET.split(":"))
            if (_et_entry.hour * 60 + _et_entry.minute) >= (_cut_h * 60 + _cut_m):
                log.info("[CYCLE %d] %s SKIP near_eod_cutoff (et=%s >= %s)",
                         cycle, sym, _et_entry.strftime("%H:%M"), NO_ENTRY_AFTER_ET)
                continue
        except Exception as _cut_ex:
            # Fail-OPEN: never let a clock parse error halt trading or block a sell.
            log.warning("[CYCLE %d] near_eod_cutoff check error (fail-open): %s",
                        cycle, _cut_ex)
        # A7: market_sense composite gate — market hours, lunch chop,
        # SPY drift, force-flat window. Earnings stub.
        _ms = _market_sense_allow_entry(sym, spy_pct_today=_spy_pct_today)
        if not _ms.allow:
            log.info("[CYCLE %d] %s skipped by market_sense: %s", cycle, sym, _ms.reason)
            continue
        # Pair-status ladder
        if sym in pair_status_blocked:
            log.info("[CYCLE %d] %s blocked by pair_status — skipping", cycle, sym)
            continue
        # PREMONDAY 2026-05-17: per-symbol re-entry cooldown after non-stop_loss
        # exit. Stop_loss handled by strike-block ladder below.
        _last_exit = (st.meta.get("last_exit_ts") or {}).get(sym, 0)
        if _last_exit and (int(time.time()) - int(_last_exit)) < REENTRY_COOLDOWN_SEC:
            _wait = REENTRY_COOLDOWN_SEC - (int(time.time()) - int(_last_exit))
            log.info("[CYCLE %d] %s cooldown_after_exit — %ds remaining", cycle, sym, _wait)
            continue
        # MIN_SCORE_TO_TRADE — applies to LEGACY only. Trader-primary entries
        # have their own per-symbol confidence floor in alpaca_trader.decide_entry.
        _is_trader_primary = bool(getattr(signal, "_trader_primary", False))
        if not _is_trader_primary and min_score > 0 \
                and float(getattr(signal, "score", 0.0)) < min_score:
            log.info("[CYCLE %d] %s skipped: score %.0f < min_score %.0f",
                     cycle, sym, getattr(signal, "score", 0.0), min_score)
            continue
        # Stop-loss strike block
        if sym in st.blocked_until and cycle < st.blocked_until[sym]:
            log.info("[CYCLE %d] %s blocked until cycle %d (stop_loss strikes) — skipping",
                     cycle, sym, st.blocked_until[sym])
            continue
        elif sym in st.blocked_until and cycle >= st.blocked_until[sym]:
            st.blocked_until.pop(sym, None)
            st.stop_loss_strikes.pop(sym, None)

        # Sizing — trader-primary uses pre-computed size_mult from decide_entry;
        # legacy uses 1.0 (trader pre-pass already produced the trader-primary path).
        if _is_trader_primary:
            _detail = getattr(signal, "_trader_detail", {}) or {}
            _brain_size_mult = float(_detail.get("size_mult", 1.0))
            _brain_reason = str(_detail.get("reason", ""))
        else:
            _brain_size_mult = 1.0
            _brain_reason = ""

        MIN_TRADE_USD = 30.0
        trade_usd = max(MIN_TRADE_USD,
                        trade_size_override * size_mult * _brain_size_mult)
        available_cash = cash - CASH_RESERVE_USD - (open_count * trade_usd)
        if available_cash < trade_usd:
            break

        _full_reason = (signal.reason if not _brain_reason
                        else (signal.reason + " | " + _brain_reason))
        log.info("[CYCLE %d] BUY %s @ $%.2f | rsi=%.1f reason=%s (size=$%.0f sup=%s)",
                 cycle, sym, snap.price, snap.rsi, _full_reason, trade_usd, sup_mode)
        fill = buy_notional(sym, trade_usd)
        if isinstance(fill, dict) and fill.get("error") == "account_blocked":
            _mark_account_blocked(st, account, cycle)
            log.error("[CYCLE %d] [HALT] %s buy denied — ACCOUNT_BLOCKED; halting cycle", cycle, sym)
            save_state(st)
            return
        if fill:
            fill_price = float(fill.get("filled_avg_price") or snap.price)
            record_buy(st, sym, fill_price, trade_usd)
            if sym in st.positions:
                st.positions[sym]._entry_signal = signal.reason
                st.positions[sym].entry_score = float(getattr(signal, "score", 0.0) or 0.0)  # D-045: carry score for trade-log attribution
            # Step 2 (Alpaca→Kraken parity): capture entry meta into st.meta
            # for downstream auto-tune / exit ledger / sentinel consumption.
            try:
                _trader_cls = getattr(signal, "_trader_classifier", None) or {}
                _cls_state = str(_trader_cls.get("state", "") or "")
                _cls_conf  = float(_trader_cls.get("confidence", 0.0) or 0.0)
                st.meta.setdefault("entry_classifier_state", {})[sym] = _cls_state
                st.meta.setdefault("entry_classifier_conf",  {})[sym] = _cls_conf
                st.meta.setdefault("entry_rsi",              {})[sym] = float(getattr(snap, "rsi", 0.0) or 0.0)
                st.meta.setdefault("entry_regime",           {})[sym] = str(st.pair_regime.get(sym, ""))
                st.meta.setdefault("entry_score",            {})[sym] = float(getattr(signal, "score", 0.0) or 0.0)
            except Exception:
                pass
            open_count += 1
            try:
                import sys as _sys
                if r"C:\Projects\supervisor" not in _sys.path:
                    _sys.path.insert(0, r"C:\Projects\supervisor")
                from supervisor_execution import log_execution
                log_execution("alpaca", sym, "BUY", trade_usd, fill_price, 0.0, signal.reason)
            except Exception:
                log.warning("[EXEC_LOG] log_execution failed bot=alpaca side=BUY sym=%s", sym, exc_info=True)

    # ── Outcome analyzer — run every 30 cycles (~30 min at 60s cycle) ─
    if cycle % 30 == 0 and cycle > 0:
        try:
            from alpaca_outcome_analyzer import run_analyzer as _run_oa
            _run_oa()
        except Exception:
            log.warning("[OUTCOME] analyzer run failed", exc_info=True)

    # ── Brain — self-tune parameters every 10 cycles ────────────────
    if cycle % 10 == 0:
        positions_str = ", ".join(
            f"{sym}@${pos.entry_price:.2f}" for sym, pos in st.positions.items()
        ) or "none"
        # Expose live equity/peak on state so brain can read them (legacy)
        st.equity = equity
        # Local-first: skip brain API call when entries are blocked (zero value)
        # D-034: also gate when account is blocked OR there have been no REAL fills
        # in 2 days — never auto-tune (e.g. increase size) on frozen/fake data.
        _brain_blocked = bool(st.meta.get("account_blocked"))
        _brain_stale = _no_recent_real_fills(st, max_age_days=2)
        if entry_ok and not _brain_blocked and not _brain_stale:
            brain_run(st, cycle, positions_str)
        elif cycle % 100 == 0:
            log.info("[BRAIN] Skipped — entry_ok=%s account_blocked=%s stale_data=%s "
                     "(no param changes)", entry_ok, _brain_blocked, _brain_stale)

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
                # Before sleeping, refresh canonical state fields once per
                # closed session (ALPACA_STATE_SCHEMA_UNIFY). _run_cycle
                # won't be invoked during closed hours, so without this the
                # state file would freeze at its last open-hours snapshot.
                try:
                    _acct = get_account()
                    if _acct is not None:
                        _eq = float(_acct.equity)
                        _un = float(getattr(_acct, "unrealized_pl", 0) or 0)
                        if _eq > st.peak_equity:
                            st.peak_equity = _eq
                        _peak = st.peak_equity if st.peak_equity > 0 else _eq
                        _dd = (_eq - _peak) / _peak * 100 if _peak > 0 else 0.0
                        st.equity_usd = _eq
                        st.unrealized_pnl_usd = _un
                        st.dd_pct = _dd
                        st.cycle = cycle
                        save_state(st)
                except Exception as _exc:
                    log.warning("[CLOSED] canonical state refresh failed: %s", _exc)

                # After-hours monitor — run once per closed session
                if not after_hours_logged:
                    _after_hours_monitor()
                    after_hours_logged = True

                wake_buffer = 300  # wake 5 min before open
                sleep_secs  = max(60, secs_to_open - wake_buffer)
                wake_time   = datetime.now(timezone.utc) + timedelta(seconds=sleep_secs)
                log.info(
                    "Market closed. Next open in %.1fh — sleeping until %s (eq=$%.2f dd=%.2f%%)",
                    secs_to_open / 3600,
                    wake_time.strftime("%H:%M UTC"),
                    st.equity_usd, st.dd_pct,
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
