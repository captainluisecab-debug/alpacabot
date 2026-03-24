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
from typing import Dict, Optional

log = logging.getLogger("alpaca_state")

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpaca_state.json")


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
        )
        for sym, p in (raw.get("positions") or {}).items():
            st.positions[sym] = BotPosition(**p)
        return st
    except Exception as exc:
        log.error("Failed to load state: %s — starting fresh", exc)
        return BotState()


def save_state(st: BotState) -> None:
    raw = {
        "realized_pnl_usd": st.realized_pnl_usd,
        "total_trades": st.total_trades,
        "winning_trades": st.winning_trades,
        "losing_trades": st.losing_trades,
        "cycle": st.cycle,
        "peak_equity": st.peak_equity,
        "positions": {sym: asdict(p) for sym, p in st.positions.items()},
        "stop_loss_strikes": dict(st.stop_loss_strikes),
        "blocked_until": dict(st.blocked_until),
        "breakeven_armed": sorted(st.breakeven_armed),
    }
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
    except Exception as exc:
        log.error("Failed to save state: %s", exc)


def record_buy(st: BotState, symbol: str, entry_price: float, usd: float) -> None:
    st.positions[symbol] = BotPosition(
        symbol=symbol,
        entry_price=entry_price,
        entry_ts=int(time.time()),
        usd_invested=usd,
    )
    save_state(st)


def record_sell(st: BotState, symbol: str, exit_price: float, reason: str = "") -> float:
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
    save_state(st)
    return pnl
