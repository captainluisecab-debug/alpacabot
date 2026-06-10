"""
alpaca_reconcile.py — safe orphan adoption + phantom handling (D-034 root-cause fix).

THE BUG it fixes: the old reconcile (alpaca_engine.py) DROPPED locals-not-on-exchange
but REFUSED to adopt exchange-positions-not-local (orphan_recovery disabled, D-034),
leaving real positions permanently unmanaged. D-034 disabled adoption because the OLD
code FABRICATED entry prices -> thrash + poisoned tuner.

THE FIX: Alpaca's Position carries avg_entry_price (Alpaca's own real VWAP), so adoption
uses REAL exchange truth and NEVER fabricates a price. The thrash failure mode is
structurally removed. A recent-fill VWAP walk is used ONLY as a non-fatal corroboration.
If avg_entry_price is ever missing AND fills are unusable -> FLAG for manual review,
NEVER guess.

Runs PER-CYCLE inside the engine reconcile block, so an orphan can never silently exist
for more than one cycle. Idempotent. The caller (engine) handles the API-error case
(get_positions() -> None -> skip reconcile); this module is only entered with a genuine
live_positions dict (a real {} means genuinely flat).

Scope: reconcile/adoption ONLY. Does NOT touch the stale_data brain-gate, classifier,
strategy, trader, DD guard, or supervisor wiring.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict

import alpaca_broker
from alpaca_state import BotPosition

_HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT_PATH = os.path.join(_HERE, "alpaca_reconcile_audit.jsonl")

MIN_ADOPT_NOTIONAL = 1.0      # below this $ notional -> dust, do not adopt (left on exchange)
CORROBORATE_TOL_PCT = 2.0     # fill-walk vs avg_entry_price agreement tolerance (log only)
SUSPICIOUS_AGE_SEC = 300      # a local position dropped within 5 min of entry is suspicious


def _audit(rec: dict) -> None:
    try:
        with open(AUDIT_PATH, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def _flag(msg: str, log) -> None:
    """Surface a HIGH concern. Always logs; best-effort escalation (never raises)."""
    log.warning("[RECONCILE][FLAG] %s", msg)
    try:
        from escalation_client import write_escalation
        write_escalation(severity="HIGH", source="alpaca_reconcile", message=msg)
    except Exception:
        pass


def _corroborate(symbol: str, qty: float, avg: float, log):
    """Volume-weighted avg of recent BUY fills, compared to avg_entry_price.
    Returns (status_str, recovered_entry_ts). NEVER fatal — corroboration only."""
    try:
        fills = alpaca_broker.get_recent_buy_fills(symbol)  # [(qty, avg_px, filled_at_epoch), ...] most-recent first
    except Exception as e:  # noqa: BLE001
        log.debug("[RECONCILE] corroboration fetch failed for %s: %s", symbol, e)
        fills = []
    if not fills:
        return "no_fills", 0
    acc_q = acc_c = 0.0
    latest_ts = 0
    for fq, fap, ts in fills:
        take = min(float(fq), max(0.0, qty - acc_q))
        if take <= 0:
            break
        acc_c += take * float(fap)
        acc_q += take
        latest_ts = max(latest_ts, int(ts or 0))
        if acc_q >= qty * 0.99:
            break
    if acc_q <= 0:
        return "no_buy_fills", 0
    vwap = acc_c / acc_q
    if avg > 0 and abs(vwap - avg) / avg * 100.0 <= CORROBORATE_TOL_PCT:
        return f"corroborated(vwap=${vwap:.2f})", latest_ts
    return f"diverged(vwap=${vwap:.2f}_vs_avg=${avg:.2f})", latest_ts


def _adopt_one(st, sym: str, live, cycle: int, log) -> bool:
    """Adopt one exchange position into local state using REAL avg_entry_price.
    Returns True if adopted. Never fabricates a price."""
    try:
        qty = float(getattr(live, "qty", 0) or 0)
        avg = float(getattr(live, "avg_entry_price", 0) or 0)
        cost_basis = float(getattr(live, "cost_basis", 0) or 0)
        current = float(getattr(live, "current_price", 0) or 0)
    except Exception as e:  # noqa: BLE001
        _flag(f"{sym}: could not parse live position ({e}) — NOT adopting", log)
        return False

    notional = qty * (current or avg)
    if qty <= 0 or notional < MIN_ADOPT_NOTIONAL:
        log.info("[SYNC][DUST] %s qty=%.6f notional=$%.2f < $%.2f — not adopting (left on exchange)",
                 sym, qty, notional, MIN_ADOPT_NOTIONAL)
        _audit({"ts": int(time.time()), "cycle": cycle, "sym": sym, "action": "DUST_SKIP",
                "qty": qty, "notional": round(notional, 2)})
        return False

    if avg <= 0:
        # NEVER fabricate. Try fills purely to report; either way, do not adopt — flag.
        status, _ = _corroborate(sym, qty, 0.0, log)
        _flag(f"{sym}: avg_entry_price missing/zero (fills: {status}) — NOT adopting, manual review", log)
        _audit({"ts": int(time.time()), "cycle": cycle, "sym": sym, "action": "FLAG_NO_BASIS", "fills": status})
        return False

    status, rec_ts = _corroborate(sym, qty, avg, log)
    entry_ts = rec_ts if rec_ts > 0 else int(time.time())
    usd_invested = cost_basis if cost_basis > 0 else qty * avg
    st.positions[sym] = BotPosition(
        symbol=sym, entry_price=avg, entry_ts=entry_ts,
        usd_invested=usd_invested, _entry_signal="adopted", peak_pnl_pct=0.0,
    )
    st.breakeven_armed.discard(sym)  # let the engine SELL loop re-derive trail/stop from live P&L
    log.warning("[SYNC][ADOPT] %s adopted: entry=$%.2f (avg_entry_price) qty=%.6f invested=$%.2f "
                "ts=%d corroboration=%s", sym, avg, qty, usd_invested, entry_ts, status)
    _audit({"ts": int(time.time()), "cycle": cycle, "sym": sym, "action": "ADOPTED",
            "entry_price": avg, "qty": qty, "usd_invested": round(usd_invested, 2), "corroboration": status})
    return True


def reconcile_positions(st, live_positions: Dict, cycle: int, log) -> None:
    """Per-cycle two-way reconcile. Caller guarantees live_positions is a genuine dict
    (None/API-error handled upstream). A real {} means genuinely flat."""
    # (a) DROP locals the exchange no longer shows (benign: external/own close succeeded).
    #     FLAG (not FATAL — per-cycle reconcile) if the dropped position is suspiciously young.
    for sym in list(st.positions.keys()):
        if sym not in live_positions:
            pos = st.positions[sym]
            age = int(time.time()) - int(getattr(pos, "entry_ts", 0) or 0)
            if 0 < age < SUSPICIOUS_AGE_SEC:
                _flag(f"{sym}: local position only {age}s old dropped (exchange shows flat) — suspicious "
                      f"(fill-not-saved / runaway-sell precursor); review", log)
            log.warning("[SYNC] %s not in Alpaca positions — removing from local state (age=%ds)", sym, age)
            st.positions.pop(sym, None)
            st.breakeven_armed.discard(sym)
            _audit({"ts": int(time.time()), "cycle": cycle, "sym": sym, "action": "DROP_LOCAL", "age_sec": age})

    # (b) ADOPT exchange positions not in local state — SAFE (real avg_entry_price, never fabricated).
    for sym, live in live_positions.items():
        if sym not in st.positions:
            _adopt_one(st, sym, live, cycle, log)
