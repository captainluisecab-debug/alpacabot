"""
escalation_client.py — Bot-side escalation client.

Shared by all bots. Each bot's local Sonnet brain calls this to:
  1. Detect roadblocks it cannot resolve locally
  2. Write an escalation request to Opus (via supervisor)
  3. Read and apply Opus's response

Opus qualities the bot should expect in responses:
  - Clear decisive action, not vague advice
  - May override the bot's parameters directly
  - May disagree with the bot and explain why
  - May spot opportunities the bot hasn't seen
  - Response expires after 30 min if not consumed
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("escalation_client")

SUPERVISOR_DIR  = r"C:\Projects\supervisor"
ESCALATION_DIR  = os.path.join(SUPERVISOR_DIR, "escalations")
RESPONSE_EXPIRY = 1800  # 30 minutes

os.makedirs(ESCALATION_DIR, exist_ok=True)


# ── Roadblock definitions ─────────────────────────────────────────────

ROADBLOCKS = {
    "CONSECUTIVE_BLOCKS":    {"urgency": "HIGH",   "threshold_cycles": 50},
    "CONSECUTIVE_LOSSES":    {"urgency": "HIGH",   "threshold_trades": 3},
    "ALL_ADX_BLOCKED":       {"urgency": "HIGH",   "threshold_cycles": 30},
    "SCORE_CONFUSION":       {"urgency": "MEDIUM", "threshold_pct": 80},
    "SUPERVISOR_DISAGREE":   {"urgency": "MEDIUM"},
    "PARAM_CONFLICT":        {"urgency": "MEDIUM"},
    "OPPORTUNITY_SPOTTED":   {"urgency": "LOW"},
    "BRAIN_UNCERTAIN":       {"urgency": "LOW",    "threshold_cycles": 5},
}


# ── Roadblock detectors ───────────────────────────────────────────────

class RoadblockDetector:
    """
    Stateful detector — instantiate once per bot brain, persist across cycles.
    """

    def __init__(self, bot: str):
        self.bot = bot
        self._consec_blocks   = 0
        self._consec_losses   = 0
        self._all_adx_cycles  = 0
        self._brain_uncertain = 0
        self._last_escalation: dict[str, float] = {}
        self._cooldown_sec = 1800  # 30 min per problem code

    def _on_cooldown(self, code: str) -> bool:
        return time.time() - self._last_escalation.get(code, 0) < self._cooldown_sec

    def _mark(self, code: str):
        self._last_escalation[code] = time.time()

    def tick_blocked(self, all_adx_blocked: bool = False):
        """Call each cycle when no entry was made."""
        self._consec_blocks += 1
        if all_adx_blocked:
            self._all_adx_cycles += 1
        else:
            self._all_adx_cycles = 0

    def tick_entry(self):
        """Call when a successful entry is made."""
        self._consec_blocks  = 0
        self._all_adx_cycles = 0

    def tick_loss(self):
        """Call when a trade closes at a loss."""
        self._consec_losses += 1

    def tick_win(self):
        """Call when a trade closes at a profit."""
        self._consec_losses = 0

    def detect(self, context: dict) -> Optional[dict]:
        """
        Check all roadblock conditions. Returns first detected roadblock dict,
        or None if all clear.
        Context should include: scores, adx_values, recent_losses, supervisor_cmd, etc.
        """
        # ALL_ADX_BLOCKED — all pairs blocked specifically by ADX threshold
        if (self._all_adx_cycles >= ROADBLOCKS["ALL_ADX_BLOCKED"]["threshold_cycles"]
                and not self._on_cooldown("ALL_ADX_BLOCKED")):
            self._mark("ALL_ADX_BLOCKED")
            adx_vals = context.get("adx_values", {})
            return {
                "problem_code": "ALL_ADX_BLOCKED",
                "urgency": "HIGH",
                "context": {
                    **context,
                    "adx_block_cycles": self._all_adx_cycles,
                    "adx_values": adx_vals,
                },
                "question": (
                    f"All {len(adx_vals)} pairs have been ADX-blocked for "
                    f"{self._all_adx_cycles} of my cycles (~{self._all_adx_cycles} min). "
                    f"ADX values: {adx_vals}. Current threshold: {context.get('adx_threshold', 15)}. "
                    f"Market appears to be in low-trend consolidation. "
                    f"Should I lower the ADX threshold? If yes, to what value? "
                    f"Or should I wait for a breakout? What do you see from portfolio level?"
                ),
            }

        # CONSECUTIVE_BLOCKS — general entry drought
        if (self._consec_blocks >= ROADBLOCKS["CONSECUTIVE_BLOCKS"]["threshold_cycles"]
                and not self._on_cooldown("CONSECUTIVE_BLOCKS")):
            self._mark("CONSECUTIVE_BLOCKS")
            return {
                "problem_code": "CONSECUTIVE_BLOCKS",
                "urgency": "HIGH",
                "context": {**context, "consec_blocks": self._consec_blocks},
                "question": (
                    f"I've been unable to open a new position for {self._consec_blocks} cycles. "
                    f"Scores: {context.get('top_scores', {})}. "
                    f"Block reasons: {context.get('block_reasons', [])}. "
                    f"Is this market-wide or am I too restrictive? "
                    f"What should I adjust to start finding entries again?"
                ),
            }

        # CONSECUTIVE_LOSSES
        if (self._consec_losses >= ROADBLOCKS["CONSECUTIVE_LOSSES"]["threshold_trades"]
                and not self._on_cooldown("CONSECUTIVE_LOSSES")):
            self._mark("CONSECUTIVE_LOSSES")
            return {
                "problem_code": "CONSECUTIVE_LOSSES",
                "urgency": "HIGH",
                "context": {**context, "consec_losses": self._consec_losses},
                "question": (
                    f"I've had {self._consec_losses} consecutive losing trades. "
                    f"Win rate: {context.get('win_rate', 0):.0f}%. "
                    f"Recent actions: {context.get('recent_actions', [])}. "
                    f"Should I pause entries, tighten parameters, or is this normal variance? "
                    f"What corrective action do you recommend?"
                ),
            }

        return None

    def flag_supervisor_disagree(self, supervisor_cmd: dict,
                                  local_score: float, local_reasoning: str) -> dict:
        """
        Bot disagrees with supervisor command based on local data.
        Returns escalation dict to be written immediately.
        """
        return {
            "problem_code": "SUPERVISOR_DISAGREE",
            "urgency": "MEDIUM",
            "disagrees_with_supervisor": True,
            "context": {
                "supervisor_command": supervisor_cmd,
                "local_top_score": local_score,
            },
            "local_reasoning": local_reasoning,
            "question": (
                f"You commanded {supervisor_cmd.get('mode')} at {supervisor_cmd.get('size_mult')}x, "
                f"but locally I'm seeing a score of {local_score:.1f} with "
                f"{local_reasoning}. "
                f"I want to flag this for your review. Should I follow your command or "
                f"do you want to authorize a targeted entry given my local data?"
            ),
        }

    def flag_opportunity(self, signal: str, data: dict) -> dict:
        """Bot spotted something positive — proactive alert to Opus."""
        return {
            "problem_code": "OPPORTUNITY_SPOTTED",
            "urgency": "LOW",
            "context": data,
            "question": (
                f"I spotted a potential opportunity: {signal}. "
                f"Data: {json.dumps(data)}. "
                f"I'm not in NORMAL mode currently. Should I act on this? "
                f"If yes, authorize an override for this specific setup."
            ),
        }


# ── Request writer ────────────────────────────────────────────────────

def write_escalation(bot: str, escalation: dict):
    """Write an escalation request for Opus to pick up."""
    req_path = os.path.join(ESCALATION_DIR, f"{bot}_request.json")

    # Don't overwrite a pending request that hasn't been handled yet
    if os.path.exists(req_path):
        try:
            age = time.time() - os.path.getmtime(req_path)
            if age < 600:  # respect existing request for 10 min
                log.debug("[ESCALATION] Request already pending for %s — skipping", bot)
                return
        except Exception:
            pass

    payload = {
        "bot":    bot,
        "ts":     datetime.now(timezone.utc).isoformat(),
        **escalation,
    }
    try:
        with open(req_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log.info("[ESCALATION] %s escalated: [%s] %s",
                 bot.upper(), escalation.get("urgency"), escalation.get("problem_code"))
    except Exception as exc:
        log.error("[ESCALATION] Failed to write request: %s", exc)


# ── Response reader ───────────────────────────────────────────────────

def read_response(bot: str) -> Optional[dict]:
    """
    Check if Opus has responded. Returns response dict or None.
    Deletes the response file after reading (one-shot).
    """
    res_path = os.path.join(ESCALATION_DIR, f"{bot}_response.json")
    if not os.path.exists(res_path):
        return None

    try:
        with open(res_path, encoding="utf-8") as f:
            response = json.load(f)
    except Exception as exc:
        log.error("[ESCALATION] Cannot read response: %s", exc)
        return None

    # Check expiry
    ts_str = response.get("ts", "")
    if ts_str:
        try:
            from datetime import datetime, timezone
            ts = datetime.fromisoformat(ts_str)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age > RESPONSE_EXPIRY:
                log.warning("[ESCALATION] Response for %s expired (%ds old) — discarding", bot, age)
                os.remove(res_path)
                return None
        except Exception:
            pass

    # Consume — delete after reading
    try:
        os.remove(res_path)
    except Exception:
        pass

    log.info("[ESCALATION] %s received Opus response: %s | negotiation=%s",
             bot.upper(), response.get("decision", "")[:60],
             response.get("negotiation_outcome", "N/A"))
    log.info("[ESCALATION] Opus to %s: %s",
             bot.upper(), response.get("message_to_bot", "")[:100])

    return response


# ── Apply response in bot brain ───────────────────────────────────────

def apply_response(response: dict, current_overrides: dict,
                   param_bounds: dict) -> dict:
    """
    Apply Opus response actions to the bot's local parameter overrides.
    Returns updated overrides dict.
    """
    if not response:
        return current_overrides

    new_overrides = dict(current_overrides)
    actions = response.get("actions", [])

    for action in actions:
        atype = action.get("type", "")

        if atype == "adjust_param":
            param = action.get("param", "")
            value = action.get("value")
            if param in param_bounds and value is not None:
                lo, hi = param_bounds[param]
                value = max(lo, min(hi, float(value)))
                new_overrides[param] = value
                log.info("[ESCALATION] Applied: %s = %s (from Opus)", param, value)

        elif atype == "strategic_directive":
            stance = action.get("stance", "")
            hours  = action.get("hours", 1)
            log.info("[ESCALATION] Strategic directive from Opus: %s for %sh | %s",
                     stance, hours, action.get("reason", ""))
            # Store directive for bot to act on
            new_overrides["_opus_directive"] = stance
            new_overrides["_opus_directive_expires"] = time.time() + hours * 3600

        elif atype in ("override_mode", "confirm_supervisor",
                       "opportunity_alert", "capital_reallocation",
                       "escalate_to_human"):
            # These are handled supervisor-side — just log
            log.info("[ESCALATION] Supervisor-side action applied: %s", atype)

    return new_overrides
