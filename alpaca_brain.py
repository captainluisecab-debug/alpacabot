"""
alpaca_brain.py — Local self-adaptation brain for alpacabot.
Runs every BRAIN_EVERY_CYCLES cycles. Uses Claude to tune strategy parameters.
Writes overrides to alpaca_brain_overrides.json which engine reads each cycle.
"""
from __future__ import annotations
import json, os, logging, datetime, time
from typing import Optional

log = logging.getLogger("alpaca_brain")

BRAIN_EVERY_CYCLES = 10
OVERRIDES_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpaca_brain_overrides.json")
DECISIONS_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpaca_brain_decisions.jsonl")

PARAM_BOUNDS = {
    "STOP_LOSS_PCT":   (2.5, 8.0),
    "TAKE_PROFIT_PCT": (5.0, 15.0),
    "TRADE_SIZE_USD":  (30.0, 200.0),  # Floor raised: below $30 wins can't cover slippage
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

    # Deterministic rule engine — no API call, zero cost
    params = dict(current or {"STOP_LOSS_PCT": 5.0, "TAKE_PROFIT_PCT": 10.0, "TRADE_SIZE_USD": 120.0, "MAX_POSITIONS": 6})
    changes = {}
    if dd_pct > 8:
        changes["TRADE_SIZE_USD"] = max(params.get("TRADE_SIZE_USD", 80.0) * 0.75, PARAM_BOUNDS["TRADE_SIZE_USD"][0])
        changes["MAX_POSITIONS"] = max(int(params.get("MAX_POSITIONS", 4)) - 1, PARAM_BOUNDS["MAX_POSITIONS"][0])
        reasoning = f"Drawdown {dd_pct:.1f}% > 8%: reduced trade size and max positions"
    elif win_rate < 30 and state.total_trades >= 10:
        changes["TRADE_SIZE_USD"] = max(params.get("TRADE_SIZE_USD", 80.0) * 0.90, PARAM_BOUNDS["TRADE_SIZE_USD"][0])
        reasoning = f"Win rate {win_rate:.0f}% < 30% over {state.total_trades} trades: slight size reduction"
    elif win_rate >= 40 and dd_pct < 5 and state.total_trades >= 5:
        # Recovery path: scale back up when winning
        cur_size = params.get("TRADE_SIZE_USD", 80.0)
        if cur_size < 80:
            changes["TRADE_SIZE_USD"] = min(cur_size * 1.15, 80.0)
            reasoning = f"Win rate {win_rate:.0f}% >= 40%, DD {dd_pct:.1f}% < 5%: scaling back up"
        elif win_rate >= 55 and dd_pct < 3 and state.total_trades >= 20:
            # Size-up gated on >=20-trade sample. On 2026-04-23 a 5W/0L
            # sample fired this branch and pushed TRADE_SIZE_USD to $200
            # over many runaway-loop iterations. Small samples are not
            # statistically meaningful enough to justify autonomous
            # sizing increases; 20 trades is a reasonable minimum for
            # win-rate to stabilize.
            changes["TRADE_SIZE_USD"] = min(cur_size * 1.10, PARAM_BOUNDS["TRADE_SIZE_USD"][1])
            reasoning = f"Win rate {win_rate:.0f}% >= 55% over {state.total_trades} trades, DD {dd_pct:.1f}% < 3%: increasing size"
        else:
            reasoning = f"Steady state (win_rate={win_rate:.0f}%, dd={dd_pct:.1f}%): holding params"
    else:
        reasoning = f"No rule triggered (win_rate={win_rate:.0f}%, dd={dd_pct:.1f}%): holding params"

    new_overrides = dict(current or {})
    for k, v in changes.items():
        if k in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[k]
            if k == "MAX_POSITIONS":
                new_overrides[k] = int(max(lo, min(hi, float(v))))
            else:
                new_overrides[k] = round(max(lo, min(hi, float(v))), 2)

    # Adaptive brain: conditionally call Opus for param refinement
    _brain_source = "local_rules"
    try:
        import sys as _sys
        if r"C:\Projects\supervisor" not in _sys.path:
            _sys.path.insert(0, r"C:\Projects\supervisor")
        from adaptive_brain import _should_review, review_sleeve, apply_recommendations, log_review
        _review_state = {
            "last_review_ts": globals().get("_alp_last_review_ts", 0),
            "recent_win_rate": win_rate,
            "overall_win_rate": win_rate,
            "dd_trend": -dd_pct if dd_pct > 0 else 0,
            "regime_changed_since_review": False,
            "trades_since_review": state.total_trades - globals().get("_alp_trades_at_review", 0),
        }
        _do_review, _trigger = _should_review(_review_state)
        if _do_review:
            _oa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpaca_score_adjustments.json")
            _outcome = json.load(open(_oa_path)) if os.path.exists(_oa_path) else {}
            _result = review_sleeve(
                sleeve_name="alpaca",
                current_params=new_overrides,
                hard_bounds=PARAM_BOUNDS,
                outcome_summary=_outcome,
                recent_trades=[],
                market_context={"dd_pct": dd_pct, "win_rate": win_rate},
                portfolio_context={"equity": equity, "positions": positions_summary},
            )
            if _result and _result.get("action") == "adjust":
                new_overrides, _changes = apply_recommendations(new_overrides, _result, PARAM_BOUNDS)
                log_review("alpaca", _trigger, _result, _changes)
                if _changes:
                    reasoning += f" | opus: {_result.get('reasoning','')}"
                    _brain_source = "local_rules+opus"
            globals()["_alp_last_review_ts"] = time.time()
            globals()["_alp_trades_at_review"] = state.total_trades
    except Exception as _exc:
        log.warning("[BRAIN] Adaptive review failed: %s", _exc)

    # Audit trail — record every brain decision, including no-change
    try:
        with open(DECISIONS_FILE, "a", encoding="utf-8") as _f:
            _f.write(json.dumps({
                "ts":         datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "cycle":      cycle,
                "old_params": current or {},
                "new_params": new_overrides,
                "reasoning":  reasoning,
                "source":     _brain_source,
            }) + "\n")
    except Exception as _e:
        log.warning("alpaca_brain_decisions write failed: %s", _e)

    if not changes:
        log.info("[BRAIN] cycle=%d no parameter changes needed | %s", cycle, reasoning)
        return None

    # Autonomy guard — per-param pre-write check + attribution logging.
    # Drops any changes that violate rate limit, oscillation, or freeze.
    try:
        import sys as _sys
        if r"C:\Projects\supervisor" not in _sys.path:
            _sys.path.insert(0, r"C:\Projects\supervisor")
        from autonomy_guard import pre_write_check as _ag_pre, record_write as _ag_rec
        equity_val = float(getattr(state, "equity", 0) or 0)
        realized   = float(getattr(state, "realized_pnl_usd", 0) or 0)
        regime     = None  # alpaca has pair-level regime, no single dominant
        _allowed_overrides = dict(current or {})
        for k, new_v in changes.items():
            if k in PARAM_BOUNDS:
                lo, hi = PARAM_BOUNDS[k]
                clamped = max(lo, min(hi, float(new_v)))
                if k == "MAX_OPEN_POSITIONS":
                    clamped = int(clamped)
                before_v = float((current or {}).get(k, clamped))
                ok, why = _ag_pre(
                    bot="alpaca", param=k, before=before_v, after=float(clamped),
                    hypothesis=reasoning,
                    expected_impact_usd=abs(equity_val) * 0.001,
                    equity_usd=equity_val, regime=regime,
                    trigger="alpaca_brain",
                    bypass_attribution=False,
                )
                if not ok:
                    log.warning("[BRAIN] param %s BLOCKED by autonomy_guard: %s", k, why)
                    continue
                _allowed_overrides[k] = clamped
                _ag_rec(
                    bot="alpaca", param=k, before=before_v, after=float(clamped),
                    hypothesis=reasoning,
                    expected_impact_usd=abs(equity_val) * 0.001,
                    equity_usd=equity_val, regime=regime,
                    trigger="alpaca_brain",
                    realized_pnl_t0=realized,
                )
        new_overrides = _allowed_overrides
    except Exception as _agexc:
        log.warning("[BRAIN] autonomy_guard integration failed: %s — writing unguarded", _agexc)

    save_overrides(new_overrides)
    log.info("[BRAIN] cycle=%d overrides updated: %s | %s", cycle, new_overrides, reasoning)
    return new_overrides


def check_escalations(state, cycle: int):
    """Check for Opus responses and escalate roadblocks to Opus."""
    try:
        from escalation_client import RoadblockDetector, write_escalation, read_response, apply_response
        if not hasattr(check_escalations, "_detector"):
            check_escalations._detector = RoadblockDetector("alpacabot")
        det = check_escalations._detector

        equity    = getattr(state, "equity", 0)
        peak      = getattr(state, "peak_equity", equity)
        dd_pct    = ((peak - equity) / peak * 100) if peak > 0 else 0
        win_rate  = (state.winning_trades / state.total_trades * 100) if state.total_trades > 0 else 0
        context   = {
            "cycle":     cycle,
            "equity":    equity,
            "dd_pct":    dd_pct,
            "win_rate":  win_rate,
            "trades":    state.total_trades,
            "positions": list(getattr(state, "positions", {}).keys()),
        }

        if state.total_trades > 3 and win_rate < 40:
            det.tick_loss()
        elif state.winning_trades > 0:
            det.tick_win()

        roadblock = det.detect(context)
        if roadblock:
            write_escalation("alpacabot", roadblock)

        resp = read_response("alpacabot")
        if resp:
            overrides = load_overrides()
            new_ov = apply_response(resp, overrides, PARAM_BOUNDS)
            if new_ov != overrides:
                save_overrides(new_ov)
                log.info("[ESCALATION] Applied Opus response: %s", resp.get("decision","")[:80])
    except Exception as e:
        log.debug("[ESCALATION] alpacabot client error: %s", e)
