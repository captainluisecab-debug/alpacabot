"""
test_alpaca_state_entry_score.py — D-045 offline validation for the entry_score
instrumentation (additive signal-score logging). No live API. No money. No orders.

SAFETY (2026-06-10): record_buy() internally calls save_state(), and log_trade()
appends to TRADE_LOG. This test therefore redirects BOTH S.STATE_FILE and
OA.TRADE_LOG to temp paths for the ENTIRE run, in a try/finally, so nothing can
ever write to the live alpaca_state.json / alpaca_trade_log.jsonl. (An earlier
version restored STATE_FILE before a record_buy call and clobbered live state.)

Verifies: the new BotPosition.entry_score field round-trips through save/load,
defaults to the -1.0 sentinel when absent, and reaches the closed-trade log row.
Run:  python tests/test_alpaca_state_entry_score.py
"""
from __future__ import annotations
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alpaca_state as S
import alpaca_outcome_analyzer as OA
from alpaca_state import BotState, BotPosition

_P = 0; _F = 0
def check(n, c):
    global _P, _F
    if c: _P += 1; print(f"  [PASS] {n}")
    else: _F += 1; print(f"  [FAIL] {n}")


def main():
    # Redirect ALL persistence to temp paths for the whole test (try/finally).
    tmp_state = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    tmp_log   = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
    orig_state, orig_log = S.STATE_FILE, OA.TRADE_LOG
    S.STATE_FILE = tmp_state
    OA.TRADE_LOG = tmp_log
    try:
        # 1. round-trip: entry_score survives save -> load
        st = BotState()
        st.positions["AMD"] = BotPosition(symbol="AMD", entry_price=100.0, entry_ts=123, usd_invested=80.0)
        st.positions["AMD"].entry_score = 84.0
        S.save_state(st)
        st2 = S.load_state()
        check("entry_score round-trips (84.0)", abs(st2.positions["AMD"].entry_score - 84.0) < 1e-9)

        # 2. old saved dict without the key -> default sentinel -1.0
        p = BotPosition(**{"symbol": "X", "entry_price": 1.0, "entry_ts": 1, "usd_invested": 1.0})
        check("missing entry_score -> default -1.0 (back-compat)", p.entry_score == -1.0)

        # 3. record_buy default (engine overwrites at buy). NOTE: record_buy saves to S.STATE_FILE,
        #    which is the temp path here — never the live file.
        st3 = BotState(); S.record_buy(st3, "NVDA", 200.0, 80.0)
        check("record_buy position defaults -1.0", st3.positions["NVDA"].entry_score == -1.0)

        # 4. log_trade writes entry_score into the closed-trade row (+ sentinel when omitted)
        OA.log_trade("AMD", "SELL", "adopted", "take_profit", 5.0, 100.0, 105.0, 60, entry_score=84.0)
        OA.log_trade("QQQ", "SELL", "brain", "stop_loss", -2.0, 100.0, 98.0, 60)  # omitted -> sentinel
        rows = [json.loads(l) for l in open(tmp_log).read().splitlines() if l.strip()]
        check("trade-log row carries entry_score=84.0", rows[0].get("entry_score") == 84.0)
        check("trade-log row sentinel -1.0 when omitted (back-compat)", rows[1].get("entry_score") == -1.0)
    finally:
        S.STATE_FILE, OA.TRADE_LOG = orig_state, orig_log
        for f in (tmp_state, tmp_log):
            try: os.unlink(f)
            except OSError: pass

    print(f"\n  RESULT: {_P} passed, {_F} failed")
    sys.exit(1 if _F else 0)


if __name__ == "__main__":
    main()
