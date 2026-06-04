"""
alpaca_state.py — Lightweight local state for tracking bot decisions.

Alpaca is the source of truth for real positions/cash.
This file tracks entry prices and bot metadata for signal calculations.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

log = logging.getLogger("alpaca_state")

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
STATE_FILE       = os.path.join(BASE_DIR, "alpaca_state.json")
# Unified exit-ledger file. Schema matches enzobot's exit_counterfactuals.jsonl
# so sentinel triggers (B2/B4/B6/B12) can consume it with one code path.
EXIT_LEDGER_FILE = os.path.join(BASE_DIR, "alpaca_exit_counterfactuals.jsonl")

# ── Post-exit forward-price tracking (OBSERVABILITY ONLY — no trading logic) ──
# Per _opus_plan_alpaca_postexit_tracking.md (operator-approved 2026-06-01).
# Purpose: log each symbol's price at +N cycles AFTER an exit, keyed by exit_id,
# to build the winner-killer counterfactual the breakeven_stop / stop_loss-width
# analysis needs (L-013). Pure-append events; new file; NEVER raises into the
# trade path. Schema FROZEN v1 (may add fields, may NOT rename/retype).
POSTEXIT_FILE = os.path.join(BASE_DIR, "alpaca_postexit_tracking.jsonl")
# Sample offsets expressed in ENGINE CYCLES (alpaca cycle ~ market-hours poll).
# Engine cycle is ~ a few min; these are deliberately coarse: ~+15m / ~+1h / EOD-ish.
_POSTEXIT_SAMPLE_CYCLES = (3, 12, 78)   # interpreted relative to exit cycle
_POSTEXIT_MAX_CYCLES = 80               # hard eviction cap (anti-leak)
# In-memory tracker: exit_id -> {symbol, exit_price, exit_cycle, exit_reason,
#                                 peak_pct, done_offsets:set}
_POSTEXIT_TRACKER: Dict[str, dict] = {}


def _postexit_append(record: dict) -> None:
    """Pure-append one event to POSTEXIT_FILE. Never raises."""
    try:
        with open(POSTEXIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception as exc:
        log.warning("postexit append failed: %s", exc)


def register_postexit(exit_id: str, symbol: str, exit_price: float,
                      exit_cycle: int, exit_reason: str, peak_pct: float) -> None:
    """Register an exit for forward-price tracking (called from record_sell).
    Writes an 'exit_open' event and adds to the in-memory tracker. Never raises."""
    try:
        _POSTEXIT_TRACKER[exit_id] = {
            "symbol": symbol,
            "exit_price": float(exit_price),
            "exit_cycle": int(exit_cycle),
            "exit_reason": exit_reason,
            "peak_pct": float(peak_pct),
            "done_offsets": set(),
        }
        _postexit_append({
            "schema_version": 1,
            "event": "exit_open",
            "exit_id": exit_id,
            "symbol": symbol,
            "exit_price": float(exit_price),
            "exit_cycle": int(exit_cycle),
            "exit_reason": exit_reason,
            "peak_pct_at_exit": float(peak_pct),
            "ts": time.time(),
            "ts_iso": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        log.warning("register_postexit failed: %s", exc)


def sample_postexit(cycle: int, price_lookup) -> None:
    """Called once per engine cycle AFTER snapshots are fetched. For each tracked
    exit whose sample offset is due, append a 'sample' event using the in-hand
    price (price_lookup(symbol) -> float|None). Evicts finished/expired exits.
    Pure observability; NEVER raises into the trade path."""
    try:
        for exit_id in list(_POSTEXIT_TRACKER.keys()):
            t = _POSTEXIT_TRACKER[exit_id]
            elapsed = cycle - t["exit_cycle"]
            for off in _POSTEXIT_SAMPLE_CYCLES:
                if off in t["done_offsets"]:
                    continue
                if elapsed >= off:
                    px = None
                    try:
                        px = price_lookup(t["symbol"])
                    except Exception:
                        px = None
                    pct = (((px - t["exit_price"]) / t["exit_price"] * 100.0)
                           if (px and t["exit_price"]) else None)
                    _postexit_append({
                        "schema_version": 1,
                        "event": "sample",
                        "exit_id": exit_id,
                        "symbol": t["symbol"],
                        "offset_cycles": off,
                        "elapsed_cycles": elapsed,
                        "exit_price": t["exit_price"],
                        "price": (float(px) if px else None),
                        "pct_vs_exit": (round(pct, 4) if pct is not None else None),
                        "exit_reason": t["exit_reason"],
                        "peak_pct_at_exit": t["peak_pct"],
                        "ts": time.time(),
                        "ts_iso": datetime.now(timezone.utc).isoformat(),
                    })
                    t["done_offsets"].add(off)
            # Eviction: all offsets done, or past the hard cap.
            if len(t["done_offsets"]) >= len(_POSTEXIT_SAMPLE_CYCLES) or elapsed >= _POSTEXIT_MAX_CYCLES:
                _POSTEXIT_TRACKER.pop(exit_id, None)
    except Exception as exc:
        log.warning("sample_postexit failed: %s", exc)


@dataclass
class BotPosition:
    symbol: str
    entry_price: float
    entry_ts: int
    usd_invested: float
    # PREMONDAY 2026-05-17: persisted entry_signal so it survives save/load
    # cycles. Pre-fix: runtime-monkey-patched, destroyed by asdict serializer,
    # caused 17 trades to log as "unknown" cohort (F-5 audit).
    _entry_signal: str = "unknown"
    # F-1 v3 (2026-05-20): actual peak P&L observed since entry. Engine
    # updates each cycle in the SELL loop. Replaces bars[-20:]-derived peak
    # in alpaca_strategy.py that included daily-bar highs from BEFORE entry
    # and fired false trail_stop on fresh intraday entries (TSLA/AMZN
    # 2026-05-20 09:35-09:36 evidence). Ratchets up monotonically from
    # 0.0 default; never decreases.
    peak_pnl_pct: float = 0.0


@dataclass
class BotState:
    positions: Dict[str, BotPosition] = field(default_factory=dict)
    realized_pnl_usd: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    cycle: int = 0
    stop_loss_strikes: Dict[str, int] = field(default_factory=dict)
    blocked_until: Dict[str, int] = field(default_factory=dict)
    peak_equity: float = 0.0  # high-water mark; persisted so restarts don't reset drawdown tracking
    breakeven_armed: set = field(default_factory=set)  # symbols where profit-lock is sticky-armed
    # Per-symbol regime classification (TRENDING_UP/TRENDING_DOWN/RANGING).
    # Read by supervisor so Alpaca's sleeve decisions are driven by stock
    # market state, not by crypto regime from Kraken's pair_regime.
    pair_regime: Dict[str, str] = field(default_factory=dict)
    # Canonical cross-sleeve state fields (ALPACA_STATE_SCHEMA_UNIFY).
    # Engine populates these each cycle; autonomy_guard reads them without
    # the fallback ladder. Aliases (peak_equity, realized_pnl_usd) still work.
    equity_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    dd_pct: float = 0.0
    # Per-symbol entry metadata captured on BUY (Step 2 — Alpaca→Kraken parity).
    # Holds dicts keyed by symbol: entry_classifier_state, entry_classifier_conf,
    # entry_rsi, entry_regime, entry_score. Cleaned up on close in record_sell.
    meta: Dict[str, Any] = field(default_factory=dict)


def load_state() -> BotState:
    if not os.path.exists(STATE_FILE):
        return BotState()
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        st = BotState(
            realized_pnl_usd=raw.get("realized_pnl_usd", 0.0),
            total_trades=raw.get("total_trades", 0),
            winning_trades=raw.get("winning_trades", 0),
            losing_trades=raw.get("losing_trades", 0),
            cycle=raw.get("cycle", 0),
            stop_loss_strikes={str(k): int(v) for k, v in (raw.get("stop_loss_strikes") or {}).items()},
            blocked_until={str(k): int(v) for k, v in (raw.get("blocked_until") or {}).items()},
            peak_equity=float(raw.get("peak_equity", 0.0) or 0.0),
            breakeven_armed=set(raw.get("breakeven_armed") or []),
            pair_regime={str(k): str(v) for k, v in (raw.get("pair_regime") or {}).items()},
            equity_usd=float(raw.get("equity_usd", 0.0) or 0.0),
            unrealized_pnl_usd=float(raw.get("unrealized_pnl_usd", 0.0) or 0.0),
            dd_pct=float(raw.get("dd_pct", 0.0) or 0.0),
            meta=dict(raw.get("meta") or {}),
        )
        for sym, p in (raw.get("positions") or {}).items():
            st.positions[sym] = BotPosition(**p)
        return st
    except Exception as exc:
        log.error("Failed to load state: %s — starting fresh", exc)
        return BotState()


def save_state(st: BotState) -> None:
    raw = {
        # ── Canonical cross-sleeve fields (primary) ───────────────────
        "equity_usd":         float(getattr(st, "equity_usd", 0.0) or 0.0),
        "realized_pnl_usd":   st.realized_pnl_usd,
        "unrealized_pnl_usd": float(getattr(st, "unrealized_pnl_usd", 0.0) or 0.0),
        "dd_pct":             float(getattr(st, "dd_pct", 0.0) or 0.0),
        "peak_equity_usd":    st.peak_equity,
        # ── Alpaca-specific fields ────────────────────────────────────
        "total_trades":       st.total_trades,
        "winning_trades":     st.winning_trades,
        "losing_trades":      st.losing_trades,
        "cycle":              st.cycle,
        "positions":          {sym: asdict(p) for sym, p in st.positions.items()},
        "stop_loss_strikes":  dict(st.stop_loss_strikes),
        "blocked_until":      dict(st.blocked_until),
        "breakeven_armed":    sorted(st.breakeven_armed),
        "sup_mode_since":     getattr(st, "sup_mode_since", None),
        "pair_regime":        dict(st.pair_regime),
        "meta":               dict(getattr(st, "meta", {}) or {}),
        # ── Legacy aliases (kept so existing readers don't break) ─────
        "peak_equity":        st.peak_equity,
    }
    try:
        _tmp = STATE_FILE + ".tmp"
        with open(_tmp, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
        # OSREPLACE 2026-05-09: 3-attempt retry on PermissionError to survive
        # Windows transient file-lock collisions. Mirrors enzobot/state_store.py
        # 2026-05-07 fix. Outer except preserved (silent-swallow for non-Permission
        # errors stays unchanged — separate ride-along).
        for _attempt in range(3):
            try:
                os.replace(_tmp, STATE_FILE)
                break
            except PermissionError:
                if _attempt == 2:
                    raise
                time.sleep(0.1)
    except Exception as exc:
        log.error("Failed to save state: %s", exc)


def _write_exit_ledger_row(
    *,
    symbol: str,
    entry_price: float,
    exit_price: float,
    qty: Optional[float],
    usd_invested: float,
    pnl_usd: float,
    exit_reason: str,
    hold_sec: int,
    regime_at_entry: Optional[str],
    regime_at_exit: Optional[str],
    score_at_entry: Optional[float],
    entry_classifier_state: Optional[str] = None,
    entry_classifier_conf: Optional[float] = None,
    entry_rsi: Optional[float] = None,
) -> None:
    """Append one row to alpaca_exit_counterfactuals.jsonl.

    Schema matches enzobot's exit_counterfactuals.jsonl so sentinel triggers
    (B2/B4/B6/B12) can read both files with the same parser. Brain context
    fields (entry_classifier_state/conf, entry_rsi) added by Alpaca→Kraken
    parity Step 2 — required for auto-tune bucket statistics.
    """
    now = time.time()
    # Derived qty fallback: usd_invested / entry_price
    _qty = qty if qty is not None else (usd_invested / entry_price if entry_price else 0.0)
    row = {
        "type":             "exit",
        "id":               f"{symbol}_{int(now)}",
        "ts":               now,
        "ts_iso":           datetime.now(timezone.utc).isoformat(),
        "pair":             symbol,
        "side":             "SELL",
        "entry_price":      float(entry_price),
        "exit_price":       float(exit_price),
        "qty":              float(_qty),
        "usd_invested":     float(usd_invested),
        "pnl_usd":          round(float(pnl_usd), 4),
        "exit_reason":      exit_reason,
        "hold_sec":         int(hold_sec or 0),
        "regime_at_entry":  regime_at_entry,
        "regime_at_exit":   regime_at_exit,
        "score_at_entry":   score_at_entry,
        "entry_classifier_state": entry_classifier_state,
        "entry_classifier_conf":  entry_classifier_conf,
        "entry_rsi":              entry_rsi,
        "sleeve":           "alpaca",
    }
    try:
        with open(EXIT_LEDGER_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as exc:
        log.warning("exit ledger write failed: %s", exc)


def record_buy(st: BotState, symbol: str, entry_price: float, usd: float) -> None:
    st.positions[symbol] = BotPosition(
        symbol=symbol,
        entry_price=entry_price,
        entry_ts=int(time.time()),
        usd_invested=usd,
    )
    save_state(st)


def record_sell(
    st: BotState,
    symbol: str,
    exit_price: float,
    reason: str = "",
    *,
    qty: Optional[float] = None,
    regime_at_exit: Optional[str] = None,
    score_at_entry: Optional[float] = None,
) -> float:
    pos = st.positions.pop(symbol, None)
    if pos is None:
        return 0.0
    st.breakeven_armed.discard(symbol)
    pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.usd_invested
    st.realized_pnl_usd += pnl
    st.total_trades += 1
    if pnl >= 0:
        st.winning_trades += 1
        # Profitable trade on this symbol — clear its strike record
        st.stop_loss_strikes.pop(symbol, None)
        st.blocked_until.pop(symbol, None)
    else:
        st.losing_trades += 1
        if "stop_loss" in reason.lower():
            strikes = st.stop_loss_strikes.get(symbol, 0) + 1
            st.stop_loss_strikes[symbol] = strikes
            log.info("Strike %d recorded for %s (stop_loss)", strikes, symbol)
            if strikes >= 2:
                block_until = st.cycle + 10
                st.blocked_until[symbol] = block_until
                log.warning(
                    "%s blocked until cycle %d (stop_loss strikes=%d)",
                    symbol, block_until, strikes,
                )
                # Reset strike counter so it can accumulate again after the block
                st.stop_loss_strikes[symbol] = 0

    # Snapshot brain context from st.meta BEFORE we clean it up below.
    _meta = getattr(st, "meta", {}) or {}
    _entry_state = (_meta.get("entry_classifier_state") or {}).get(symbol)
    _entry_conf  = (_meta.get("entry_classifier_conf") or {}).get(symbol)
    _entry_rsi   = (_meta.get("entry_rsi") or {}).get(symbol)
    _entry_regime = (_meta.get("entry_regime") or {}).get(symbol)
    _entry_score_meta = (_meta.get("entry_score") or {}).get(symbol)

    # Unified exit ledger (parity with enzobot for sentinel consumption)
    try:
        _hold = int(time.time()) - pos.entry_ts if pos.entry_ts else 0
        _regime_exit = regime_at_exit or st.pair_regime.get(symbol)
        _write_exit_ledger_row(
            symbol=symbol,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=qty,
            usd_invested=pos.usd_invested,
            pnl_usd=pnl,
            exit_reason=reason,
            hold_sec=_hold,
            regime_at_entry=_entry_regime,
            regime_at_exit=_regime_exit,
            score_at_entry=(score_at_entry if score_at_entry is not None else _entry_score_meta),
            entry_classifier_state=_entry_state,
            entry_classifier_conf=_entry_conf,
            entry_rsi=_entry_rsi,
        )
    except Exception as _exc:
        log.warning("[EXIT_LEDGER] write failed for %s: %s", symbol, _exc)

    # Post-exit forward-price tracking registration (OBSERVABILITY ONLY).
    # Keyed by the same exit_id format the ledger uses. Never raises.
    try:
        _peak = float(getattr(pos, "peak_pnl_pct", 0.0) or 0.0)
        register_postexit(
            exit_id=f"{symbol}_{int(time.time())}",
            symbol=symbol,
            exit_price=exit_price,
            exit_cycle=int(getattr(st, "cycle", 0) or 0),
            exit_reason=reason,
            peak_pct=_peak,
        )
    except Exception as _exc:
        log.warning("[POSTEXIT] register failed for %s: %s", symbol, _exc)

    # Clean up entry meta for closed symbol so meta dict doesn't grow unbounded
    try:
        for _k in ("entry_classifier_state", "entry_classifier_conf",
                   "entry_rsi", "entry_regime", "entry_score"):
            _d = _meta.get(_k)
            if isinstance(_d, dict):
                _d.pop(symbol, None)
    except Exception:
        pass

    save_state(st)
    return pnl
