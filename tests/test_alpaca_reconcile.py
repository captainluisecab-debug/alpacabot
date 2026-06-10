"""
test_alpaca_reconcile.py — offline tests for the D-034 orphan-fix (alpaca_reconcile.py).
Mocks the exchange (FakePos) and the fill-walk (monkeypatched). No live API. No real money.
Mirrors zerobot/tests/test_reconcile.py coverage: adopt / phantom-drop / idempotent /
dust / corroborate / no-basis-flag.
Run:  python tests/test_alpaca_reconcile.py
"""
from __future__ import annotations
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alpaca_broker
import alpaca_reconcile as R
from alpaca_state import BotState, BotPosition

log = logging.getLogger("test")
logging.basicConfig(level=logging.CRITICAL)  # silence the module's warnings during tests

_PASS = 0
_FAIL = 0


def check(name, cond):
    global _PASS, _FAIL
    if cond:
        _PASS += 1; print(f"  [PASS] {name}")
    else:
        _FAIL += 1; print(f"  [FAIL] {name}")


class FakePos:
    def __init__(self, symbol, qty, avg_entry_price, current_price=None, cost_basis=None):
        self.symbol = symbol
        self.qty = qty
        self.avg_entry_price = avg_entry_price
        self.current_price = current_price if current_price is not None else avg_entry_price
        self.cost_basis = cost_basis if cost_basis is not None else (float(qty) * float(avg_entry_price))


def _no_fills(symbol, **kw):
    return []


def _mock_fills(rows):
    return lambda symbol, **kw: rows


def main():
    orig = alpaca_broker.get_recent_buy_fills

    # T-adopt: real orphan adopted with avg_entry_price
    alpaca_broker.get_recent_buy_fills = _no_fills
    st = BotState()
    live = {"AMD": FakePos("AMD", qty=0.1205, avg_entry_price=512.65, current_price=471.9, cost_basis=61.77)}
    R.reconcile_positions(st, live, cycle=1, log=log)
    p = st.positions.get("AMD")
    check("T-adopt: AMD adopted", p is not None)
    check("T-adopt: entry_price = avg_entry_price (512.65, NOT fabricated)", p and abs(p.entry_price - 512.65) < 1e-6)
    check("T-adopt: usd_invested = cost_basis (61.77)", p and abs(p.usd_invested - 61.77) < 1e-6)
    check("T-adopt: _entry_signal = adopted", p and p._entry_signal == "adopted")
    check("T-adopt: not breakeven-armed", "AMD" not in st.breakeven_armed)

    # T-idempotent: second reconcile with same live = no re-adopt, still one position
    R.reconcile_positions(st, live, cycle=2, log=log)
    check("T-idempotent: still exactly 1 position", len(st.positions) == 1)

    # T-phantom-drop: local has a position, exchange flat -> dropped
    st2 = BotState()
    st2.positions["QQQ"] = BotPosition(symbol="QQQ", entry_price=717.0, entry_ts=1, usd_invested=100.0)
    R.reconcile_positions(st2, {}, cycle=3, log=log)
    check("T-phantom-drop: QQQ removed when exchange flat", "QQQ" not in st2.positions)

    # T-dust: tiny notional NOT adopted
    alpaca_broker.get_recent_buy_fills = _no_fills
    st3 = BotState()
    R.reconcile_positions(st3, {"TINY": FakePos("TINY", qty=0.0001, avg_entry_price=2.0, current_price=2.0)}, cycle=4, log=log)
    check("T-dust: sub-$1 notional not adopted", "TINY" not in st3.positions)

    # T-no-basis-flag: avg_entry_price = 0 -> NOT adopted, never fabricated
    st4 = BotState()
    R.reconcile_positions(st4, {"BAD": FakePos("BAD", qty=1.0, avg_entry_price=0.0, current_price=50.0)}, cycle=5, log=log)
    check("T-no-basis: avg=0 -> NOT adopted (flag, never guess)", "BAD" not in st4.positions)

    # T-corroborate: fills agree with avg -> still adopts on avg_entry_price, recovers entry_ts
    alpaca_broker.get_recent_buy_fills = _mock_fills([(0.1419, 718.00, 1748000000)])
    st5 = BotState()
    R.reconcile_positions(st5, {"QQQ": FakePos("QQQ", qty=0.1419, avg_entry_price=717.87, current_price=710.5, cost_basis=101.86)}, cycle=6, log=log)
    p5 = st5.positions.get("QQQ")
    check("T-corroborate: QQQ adopted on avg_entry_price", p5 and abs(p5.entry_price - 717.87) < 1e-6)
    check("T-corroborate: entry_ts recovered from fill", p5 and p5.entry_ts == 1748000000)

    # T-corroborate-diverge: fills disagree wildly -> STILL adopts on avg_entry_price (fills are log-only)
    alpaca_broker.get_recent_buy_fills = _mock_fills([(1.0, 999.0, 1748000000)])
    st6 = BotState()
    R.reconcile_positions(st6, {"NVDA": FakePos("NVDA", qty=1.0, avg_entry_price=200.0, current_price=205.0, cost_basis=200.0)}, cycle=7, log=log)
    p6 = st6.positions.get("NVDA")
    check("T-diverge: wrong fills CANNOT change entry_price (stays avg=200)", p6 and abs(p6.entry_price - 200.0) < 1e-6)

    alpaca_broker.get_recent_buy_fills = orig
    print(f"\n  RESULT: {_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
