"""alpaca_trader.py — Stock trader brain (decision layer).

Build: 2026-04-29 (port from kraken_trader.py for Alpaca brain wiring).

Per-symbol calibration tiered by recent performance:
  - Winners (5/5 wins recent): MSFT, NVDA, TSLA, QQQ — full conviction
  - Index/broad (SPY): conservative size, tight gates (broad market less alpha per-trade)
  - Neutral (AAPL, AMZN, AMD): moderate conviction
  - All allowed; classifier + per-symbol gates do the work

Active when policy.USE_ALPACA_BRAIN=true. Engine consults trader for entry decisions.
Score still applies as supporting input. Existing alpaca_market_sense gates run alongside.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, Any


# Universe of stocks the trader considers (matches alpaca_engine.py UNIVERSE)
ALLOWED_SYMBOLS = ('SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMD', 'AMZN')


# Classifier states that allow long entry
ALLOW_STATES = (
    'bullish_continuation',
    'breakout_long',
    'bearish_exhaustion',
)


# Classifier states that force exit on held position (regime shift)
EXIT_TRIGGER_STATES = (
    'bullish_exhaustion',
    'bearish_continuation',
    'failed_breakout',
)


# Per-symbol calibration. Tiered by recent profitability + sample size.
# Edge weights start at 1.0 for established performers; adjust as outcomes accumulate.
SYMBOL_CONFIG: Dict[str, Dict[str, float]] = {
    # WINNERS (recent 5-win streak in alpaca_state.json)
    'MSFT': {
        'size_mult_full': 1.0, 'size_mult_reduced': 0.6,
        'stop_atr_mult': 1.5, 'min_classifier_conf': 0.5, 'edge_weight': 1.0,
    },
    'NVDA': {
        'size_mult_full': 1.0, 'size_mult_reduced': 0.6,
        'stop_atr_mult': 1.5, 'min_classifier_conf': 0.5, 'edge_weight': 1.0,
    },
    'TSLA': {
        'size_mult_full': 0.7, 'size_mult_reduced': 0.4,
        'stop_atr_mult': 1.8, 'min_classifier_conf': 0.55, 'edge_weight': 0.95,  # higher vol
    },
    'QQQ': {
        'size_mult_full': 1.0, 'size_mult_reduced': 0.6,
        'stop_atr_mult': 1.5, 'min_classifier_conf': 0.5, 'edge_weight': 1.0,
    },
    # INDEX (broad market — full conviction at low vol)
    'SPY': {
        'size_mult_full': 1.0, 'size_mult_reduced': 0.6,
        'stop_atr_mult': 1.4, 'min_classifier_conf': 0.5, 'edge_weight': 1.0,
    },
    # NEUTRAL (no recent trades, moderate conviction)
    'AAPL': {
        'size_mult_full': 0.8, 'size_mult_reduced': 0.5,
        'stop_atr_mult': 1.6, 'min_classifier_conf': 0.55, 'edge_weight': 1.0,
    },
    'AMZN': {
        'size_mult_full': 0.7, 'size_mult_reduced': 0.4,
        'stop_atr_mult': 1.7, 'min_classifier_conf': 0.55, 'edge_weight': 0.95,
    },
    'AMD': {
        'size_mult_full': 0.7, 'size_mult_reduced': 0.4,
        'stop_atr_mult': 1.8, 'min_classifier_conf': 0.6, 'edge_weight': 0.9,  # higher vol
    },
}


def _open_count(positions: Dict[str, Any]) -> int:
    n = 0
    for p in positions.values():
        try:
            qty = getattr(p, 'qty', 0) or getattr(p, 'shares', 0) or 0
            if qty > 0:
                n += 1
        except Exception:
            pass
    return n


def decide_entry(symbol: str, snap, classifier_result: Optional[Dict[str, Any]],
                 cfg, positions: Dict, max_positions: int) -> Tuple[str, Any]:
    """Decide whether to enter symbol. Returns ('BUY', detail_dict) or ('SKIP', reason_str)."""
    if symbol not in ALLOWED_SYMBOLS:
        return ('SKIP', 'symbol_not_in_allowed_universe')

    if _open_count(positions) >= max_positions:
        return ('SKIP', 'max_positions_reached')

    if symbol in positions:
        return ('SKIP', 'already_holding_symbol')

    if classifier_result is None:
        return ('SKIP', 'no_classifier_data')

    state = classifier_result.get('state')
    conf = float(classifier_result.get('confidence', 0.0) or 0.0)

    if state not in ALLOW_STATES:
        return ('SKIP', 'classifier_state=' + str(state))

    sym_cfg = SYMBOL_CONFIG.get(symbol, {})
    min_conf = float(sym_cfg.get('min_classifier_conf', 0.6))
    if conf < min_conf:
        return ('SKIP', 'classifier_conf=' + ('%.2f' % conf) + '<' + ('%.2f' % min_conf))

    # Score is supporting (size boost), not gate
    score = float(getattr(snap, 'score', 0.0) or 0.0)

    # RSI cap (avoid extreme overbought even if classifier says continuation)
    rsi = float(getattr(snap, 'rsi', 0.0) or 0.0)
    if rsi > 78.0:
        return ('SKIP', 'rsi=' + ('%.1f' % rsi) + '>78 (extreme overbought)')

    # Compute size from classifier confidence + score boost + per-symbol edge
    if conf >= 0.7:
        base_mult = sym_cfg.get('size_mult_full', 0.7)
    else:
        base_mult = sym_cfg.get('size_mult_reduced', 0.4)

    if score >= 80:
        score_boost = 1.0
    elif score >= 70:
        score_boost = 0.85
    elif score >= 60:
        score_boost = 0.70
    else:
        score_boost = 0.55

    edge_weight = float(sym_cfg.get('edge_weight', 1.0))
    final_size_mult = base_mult * score_boost * edge_weight

    reason = ('alpaca_trader: cls=' + str(state) + '@' + ('%.2f' % conf)
              + ' rsi=' + ('%.1f' % rsi) + ' score=' + ('%.1f' % score)
              + ' (boost=' + ('%.2f' % score_boost) + ')'
              + ' size=' + ('%.2fx' % final_size_mult))

    return ('BUY', {
        'size_mult': final_size_mult,
        'stop_atr_mult': sym_cfg.get('stop_atr_mult', 1.7),
        'reason': reason,
        'classifier_state': state,
        'classifier_conf': conf,
    })


def decide_exit(symbol: str, snap, classifier_result: Optional[Dict[str, Any]],
                position, cfg) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Decide whether to force-exit a held position based on classifier regime shift."""
    if symbol not in ALLOWED_SYMBOLS:
        return None
    if classifier_result is None:
        return None
    if not position:
        return None
    qty = getattr(position, 'qty', 0) or getattr(position, 'shares', 0) or 0
    if qty <= 0:
        return None

    state = classifier_result.get('state')
    conf = float(classifier_result.get('confidence', 0.0) or 0.0)

    if state == 'bullish_exhaustion' and conf >= 0.6:
        return ('SELL', {'partial': False,
                         'reason': 'alpaca_trader_exit_regime_shift_to_bullish_exhaustion@' + ('%.2f' % conf)})
    if state == 'bearish_continuation' and conf >= 0.5:
        return ('SELL', {'partial': False,
                         'reason': 'alpaca_trader_exit_regime_shift_to_bearish_continuation@' + ('%.2f' % conf)})
    if state == 'failed_breakout' and conf >= 0.6:
        return ('SELL', {'partial': False,
                         'reason': 'alpaca_trader_exit_regime_shift_to_failed_breakout@' + ('%.2f' % conf)})

    return None
