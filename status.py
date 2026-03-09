"""
status.py — Quick status check for Alpaca bot.
Usage: python status.py
"""
from __future__ import annotations
import json, os

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpaca_state.json")


def main():
    # Live account data
    try:
        from alpaca_broker import get_account, get_positions
        account   = get_account()
        positions = get_positions()
        cash   = float(account.cash)   if account else 0
        equity = float(account.equity) if account else 0
    except Exception as exc:
        print(f"Could not fetch live account: {exc}")
        cash = equity = 0
        positions = {}

    # Bot state
    raw = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            raw = json.load(f)

    pnl    = raw.get("realized_pnl_usd", 0.0)
    total  = raw.get("total_trades", 0)
    wins   = raw.get("winning_trades", 0)
    losses = raw.get("losing_trades", 0)
    cycle  = raw.get("cycle", 0)
    bot_positions = raw.get("positions", {})
    wr = wins / total * 100 if total > 0 else 0

    print("=" * 55)
    print("  ALPACA SWING BOT — STATUS")
    print("=" * 55)
    print(f"  Cycle:          {cycle}")
    print(f"  Equity:         ${equity:>10.2f}")
    print(f"  Cash:           ${cash:>10.2f}")
    print()

    if positions:
        print("  Open Positions (live from Alpaca):")
        for sym, p in positions.items():
            mkt_val  = float(p.market_value)
            unrealized = float(p.unrealized_pl)
            pct      = float(p.unrealized_plpc) * 100
            entry    = bot_positions.get(sym, {}).get("entry_price", 0)
            print(f"    {sym:<6} qty={float(p.qty):.4f}  val=${mkt_val:.2f}"
                  f"  pnl=${unrealized:+.2f} ({pct:+.1f}%)"
                  f"  entry=${entry:.2f}")
    else:
        print("  Open Positions: FLAT")

    print()
    print(f"  Realized PnL:   ${pnl:>+.2f}")
    print(f"  Trades:         {total}  (W={wins} L={losses} WR={wr:.0f}%)")
    print("=" * 55)


if __name__ == "__main__":
    main()
