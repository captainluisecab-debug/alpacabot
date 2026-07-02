"""
test_account_blocked.py — unit tests for the D-034 integrity fix.
Self-contained (no pytest, no live Alpaca calls). Run: python tests/test_account_blocked.py
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alpaca_broker as B
import alpaca_engine as E

_PASS = 0
_FAIL = 0
def check(name, cond):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
        print(f"  FAIL: {name}")


class FakeAccount:
    def __init__(self, **flags):
        self.trading_blocked = flags.get("trading_blocked", False)
        self.account_blocked = flags.get("account_blocked", False)
        self.trade_suspended_by_user = flags.get("trade_suspended_by_user", False)


class FakeState:
    def __init__(self):
        self.meta = {}


def main():
    # T5 — broker classifier distinguishes account-block (40310000) from PDT (40310100)
    check("classifier: 40310000 -> True",
          B._is_account_blocked_err(Exception('{"code":40310000,"message":"new orders are rejected by user request"}')))
    check("classifier: phrase -> True", B._is_account_blocked_err(Exception("rejected by user request")))
    check("classifier: PDT 40310100 -> False", not B._is_account_blocked_err(Exception("40310100 pattern day trading")))
    check("classifier: generic -> False", not B._is_account_blocked_err(Exception("connection reset")))

    # T1 — get_positions returns None on API error (NOT {} — the thrash fix)
    _orig = B._trading_client
    try:
        B._trading_client = lambda: (_ for _ in ()).throw(RuntimeError("api down"))
        check("get_positions error -> None (not {})", B.get_positions() is None)
    finally:
        B._trading_client = _orig

    # account-flag detection
    check("acct_flags: trading_blocked", E._acct_flags_blocked(FakeAccount(trading_blocked=True)))
    check("acct_flags: trade_suspended_by_user", E._acct_flags_blocked(FakeAccount(trade_suspended_by_user=True)))
    check("acct_flags: all clear -> False", not E._acct_flags_blocked(FakeAccount()))

    # T2 — _mark_account_blocked sets flag + cooldown, escalates ONCE per episode
    _orig_esc = E._escalate_account_blocked
    calls = {"n": 0}
    try:
        E._escalate_account_blocked = lambda account, cycle: calls.__setitem__("n", calls["n"] + 1)
        st = FakeState()
        E._mark_account_blocked(st, FakeAccount(trading_blocked=True), 1)
        check("mark: flag set", st.meta.get("account_blocked") is True)
        check("mark: cooldown in future", float(st.meta.get("account_blocked_until", 0)) > time.time())
        check("mark: escalated once", calls["n"] == 1)
        E._mark_account_blocked(st, FakeAccount(trading_blocked=True), 2)  # still within cooldown
        check("mark: NOT re-escalated within episode", calls["n"] == 1)
    finally:
        E._escalate_account_blocked = _orig_esc

    # T3 — stale-data gate for the brain (the loaded-gun disarm)
    st = FakeState()
    check("stale: never sold -> False (fresh bot)", E._no_recent_real_fills(st) is False)
    st.meta["last_exit_ts"] = {"AMD": int(time.time()) - 3 * 86400}
    check("stale: 3d-old fill -> True", E._no_recent_real_fills(st, max_age_days=2) is True)
    st.meta["last_exit_ts"] = {"AMD": int(time.time()) - 3600}
    check("stale: 1h-old fill -> False", E._no_recent_real_fills(st, max_age_days=2) is False)

    print(f"\n{_PASS}/{_PASS + _FAIL} tests passed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
