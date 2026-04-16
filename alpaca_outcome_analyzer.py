"""
alpaca_outcome_analyzer.py — Closed feedback loop for Alpaca trading.

Reads alpaca_trade_log.jsonl (real trade data).
Computes per-symbol, per-signal, per-exit quality.
Writes alpaca_score_adjustments.json that the engine reads each cycle.

Runs every 30 minutes from alpaca_engine.py.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List

log = logging.getLogger("alpaca_outcome_analyzer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG = os.path.join(BASE_DIR, "alpaca_trade_log.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "alpaca_score_adjustments.json")
MIN_TRADES = 3
MIN_TRADES_PER_SYMBOL = 2


def log_trade(symbol: str, side: str, entry_signal: str, exit_reason: str,
              pnl_usd: float, entry_price: float, exit_price: float,
              hold_sec: int, rsi_at_entry: float = 0, rsi_at_exit: float = 0) -> None:
    record = {
        "type": "trade",
        "ts": time.time(),
        "symbol": symbol,
        "side": side,
        "entry_signal": entry_signal,
        "exit_reason": exit_reason,
        "pnl_usd": round(pnl_usd, 4),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "hold_sec": hold_sec,
        "rsi_at_entry": round(rsi_at_entry, 1),
        "rsi_at_exit": round(rsi_at_exit, 1),
    }
    try:
        with open(TRADE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        log.error("[ANALYZER] Failed to log trade: %s", exc)


def _read_trades(lookback_days: int = 14) -> List[dict]:
    if not os.path.exists(TRADE_LOG):
        return []
    cutoff = time.time() - (lookback_days * 86400)
    trades = []
    try:
        with open(TRADE_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("type") != "trade":
                        continue
                    if r.get("ts", 0) < cutoff:
                        continue
                    trades.append(r)
                except Exception:
                    continue
    except Exception as exc:
        log.error("[ANALYZER] Failed to read trades: %s", exc)
    return trades


def _symbol_quality(trades: List[dict]) -> Dict[str, dict]:
    symbols: Dict[str, dict] = {}
    for t in trades:
        sym = t.get("symbol", "?")
        if sym not in symbols:
            symbols[sym] = {"count": 0, "pnl": 0.0, "wins": 0}
        symbols[sym]["count"] += 1
        symbols[sym]["pnl"] += t.get("pnl_usd", 0)
        if t.get("pnl_usd", 0) > 0:
            symbols[sym]["wins"] += 1

    result = {}
    for sym, s in symbols.items():
        wr = s["wins"] / max(1, s["count"])
        avg = s["pnl"] / max(1, s["count"])
        adj = 0.0
        if s["count"] >= MIN_TRADES_PER_SYMBOL:
            if wr >= 0.6 and avg > 0:
                adj = min(2.0, avg * 0.5)
            elif wr <= 0.25 and avg < 0:
                adj = max(-2.0, avg * 0.3)
        result[sym] = {
            "count": s["count"],
            "avg_pnl": round(avg, 2),
            "win_rate": round(wr * 100, 1),
            "total_pnl": round(s["pnl"], 2),
            "score_adjustment": round(adj, 2),
        }
    return result


def _signal_quality(trades: List[dict]) -> Dict[str, dict]:
    signals: Dict[str, dict] = {}
    for t in trades:
        sig = t.get("entry_signal", "unknown")
        if sig not in signals:
            signals[sig] = {"count": 0, "pnl": 0.0, "wins": 0}
        signals[sig]["count"] += 1
        signals[sig]["pnl"] += t.get("pnl_usd", 0)
        if t.get("pnl_usd", 0) > 0:
            signals[sig]["wins"] += 1

    result = {}
    for sig, s in signals.items():
        result[sig] = {
            "count": s["count"],
            "avg_pnl": round(s["pnl"] / max(1, s["count"]), 2),
            "win_rate": round(s["wins"] / max(1, s["count"]) * 100, 1),
        }
    return result


def _exit_quality(trades: List[dict]) -> Dict[str, dict]:
    exits: Dict[str, dict] = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        if r not in exits:
            exits[r] = {"count": 0, "pnl": 0.0, "wins": 0}
        exits[r]["count"] += 1
        exits[r]["pnl"] += t.get("pnl_usd", 0)
        if t.get("pnl_usd", 0) > 0:
            exits[r]["wins"] += 1

    result = {}
    for r, s in exits.items():
        result[r] = {
            "count": s["count"],
            "avg_pnl": round(s["pnl"] / max(1, s["count"]), 2),
            "win_rate": round(s["wins"] / max(1, s["count"]) * 100, 1),
        }
    return result


def _compute_symbol_blocks(sym_q: dict) -> List[str]:
    blocks = []
    for sym, q in sym_q.items():
        if q["count"] >= MIN_TRADES and q["win_rate"] < 20 and q["avg_pnl"] < -1.0:
            blocks.append(sym)
    return blocks


def run_analyzer() -> dict:
    trades = _read_trades(lookback_days=14)
    if len(trades) < MIN_TRADES:
        log.info("[ANALYZER] Only %d trades in 14 days -- waiting for data", len(trades))
        return {}

    sym_q = _symbol_quality(trades)
    sig_q = _signal_quality(trades)
    exit_q = _exit_quality(trades)
    blocks = _compute_symbol_blocks(sym_q)

    total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
    total_wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)

    result = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lookback_days": 14,
        "total_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "overall_win_rate": round(total_wins / len(trades) * 100, 1),
        "per_symbol_quality": sym_q,
        "entry_signal_quality": sig_q,
        "exit_reason_quality": exit_q,
        "recommended_blocks": blocks,
    }

    try:
        tmp = OUTPUT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp, OUTPUT_FILE)
        log.info("[ANALYZER] Updated: %d trades, WR=%.0f%%, sym_blocks=%s",
                 len(trades), result["overall_win_rate"], blocks or "none")
    except Exception as exc:
        log.error("[ANALYZER] Failed to write: %s", exc)

    return result
