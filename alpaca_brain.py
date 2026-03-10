"""
alpaca_brain.py — Local self-adaptation brain for alpacabot.
Runs every BRAIN_EVERY_CYCLES cycles. Uses Claude to tune strategy parameters.
Writes overrides to alpaca_brain_overrides.json which engine reads each cycle.
"""
from __future__ import annotations
import json, os, logging
from typing import Optional

log = logging.getLogger("alpaca_brain")

BRAIN_EVERY_CYCLES = 10
OVERRIDES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpaca_brain_overrides.json")

PARAM_BOUNDS = {
    "STOP_LOSS_PCT":   (2.0, 8.0),
    "TAKE_PROFIT_PCT": (5.0, 15.0),
    "TRADE_SIZE_USD":  (50.0, 200.0),
    "MAX_POSITIONS":   (2, 8),
}


def load_overrides() -> dict:
    """Load current parameter overrides. Returns {} if none."""
    if not os.path.exists(OVERRIDES_FILE):
        return {}
    try:
        with open(OVERRIDES_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[BRAIN] Could not load overrides: %s", exc)
        return {}


def save_overrides(overrides: dict) -> None:
    try:
        with open(OVERRIDES_FILE, "w", encoding="utf-8") as f:
            json.dump(overrides, f, indent=2)
    except Exception as exc:
        log.warning("[BRAIN] Could not save overrides: %s", exc)


def run_brain(state, cycle: int, positions_summary: str) -> Optional[dict]:
    """
    Run brain cycle. Returns new overrides dict or None if no changes.
    Only runs every BRAIN_EVERY_CYCLES cycles.
    """
    if cycle % BRAIN_EVERY_CYCLES != 0:
        return None

    current = load_overrides()

    # Build performance summary from state
    win_rate = (state.winning_trades / state.total_trades * 100) if state.total_trades > 0 else 0
    peak_equity = getattr(state, "peak_equity", 0)
    equity = getattr(state, "equity", 0)
    dd_pct = ((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0

    prompt = f"""You are the local brain for an Alpaca stock swing trading bot.

## CURRENT STATE
- Equity: ${equity:.2f} | Peak: ${peak_equity:.2f} | DD: {dd_pct:.1f}%
- Realized PnL: ${state.realized_pnl_usd:+.2f} | Trades: {state.total_trades} | Win rate: {win_rate:.0f}%
- Open positions: {positions_summary}
- Current params: {json.dumps(current or {"STOP_LOSS_PCT": 5.0, "TAKE_PROFIT_PCT": 10.0, "TRADE_SIZE_USD": 120, "MAX_POSITIONS": 6})}

## MARKET UNIVERSE: SPY, QQQ, AAPL, MSFT, TSLA, NVDA, AMD, AMZN

## TASK
Analyze performance. Tune parameters to maximize returns while protecting capital.
If win_rate < 40%: tighten entry (reduce TRADE_SIZE_USD, tighten STOP_LOSS_PCT)
If dd > 5%: reduce TRADE_SIZE_USD and MAX_POSITIONS
If win_rate > 60% and dd < 2%: can increase TRADE_SIZE_USD slightly

Respond ONLY with valid JSON: {{"changes":{{}}, "reasoning":"one sentence"}}
Only include parameters that need changing. Empty changes={{}} means no change needed."""

    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.content[0].text.strip()
        data = json.loads(raw)
        changes = data.get("changes", {})

        if not changes:
            log.info("[BRAIN] cycle=%d no parameter changes needed | %s", cycle, data.get("reasoning", ""))
            return None

        # Apply bounds
        new_overrides = dict(current or {})
        for k, v in changes.items():
            if k in PARAM_BOUNDS:
                lo, hi = PARAM_BOUNDS[k]
                new_overrides[k] = max(lo, min(hi, float(v)))

        save_overrides(new_overrides)
        log.info("[BRAIN] cycle=%d overrides updated: %s | %s", cycle, new_overrides, data.get("reasoning", ""))
        return new_overrides

    except Exception as e:
        log.warning("[BRAIN] cycle=%d error: %s", cycle, e)
        return None
