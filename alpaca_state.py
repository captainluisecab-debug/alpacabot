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
from typing import Dict, Optional

log = logging.getLogger("alpaca_state")

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
STATE_FILE       = os.path.join(BASE_DIR, "alpaca_state.json")
# Unified exit-ledger file. Schema matches enzobot's exit_counterfactuals.jsonl
# so sentinel triggers (B2/B4/B6/B12) can consume it with one code path.
EXIT_LEDGER_FILE = os.path.join(BASE_DIR, "alpaca_exit_counterfactuals.jsonl")


@dataclass
class BotPosition:
    symbol: str
    entry_price: float
    entry_ts: int
    usd_invested: float


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
        # ── Legacy aliases (kept so existing readers don't break) ─────
        "peak_equity":        st.peak_equity,
    }
    try:
        _tmp = STATE_FILE + ".tmp"
        with open(_tmp, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
        os.replace(_tmp, STATE_FILE)
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
) -> None:
    """Append one row to alpaca_exit_counterfactuals.jsonl.

    Schema matches enzobot's exit_counterfactuals.jsonl so sentinel triggers
    (B2/B4/B6/B12) can read both files with the same parser.
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
            regime_at_entry=None,  # not currently tracked at buy time
            regime_at_exit=_regime_exit,
            score_at_entry=score_at_entry,
        )
    except Exception as _exc:
        log.warning("[EXIT_LEDGER] write failed for %s: %s", symbol, _exc)

    save_state(st)
    return pnl
