"""Alpaca path classifier - stock-calibrated 8-state classification.

Build: 2026-04-29 (port from enzobot/kraken_path_classifier.py for Alpaca brain).
Ports the same multi-indicator perception layer to stocks.

Same 8 states as Kraken: bullish_continuation | bullish_exhaustion |
bearish_continuation | bearish_exhaustion | compression | breakout_long |
breakout_short | failed_breakout | chop.

Same priority order: exhaustion > failed_breakout > breakout > continuation
> compression > chop.

Stock-vs-crypto differences (initial calibration; can be tuned from Alpaca outcomes):
  - Stocks trade ~6.5h/day vs crypto 24/7 → fewer bars per real-time period
  - Stocks generally less volatile than crypto → ATR ratios can be smaller
  - Stocks have gap-up/down at open → first 5m bar can be misleading
  - Stocks have lunch chop (handled separately by alpaca_market_sense.py)
  - SPY direction is a meta-signal (handled by alpaca_market_sense.py SPY_drift gate)

For now: thresholds identical to Kraken classifier. Calibrate from outcome data.

Pure function. No live side effects.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


STATES = (
    "bullish_continuation",
    "bullish_exhaustion",
    "bearish_continuation",
    "bearish_exhaustion",
    "compression",
    "breakout_long",
    "breakout_short",
    "failed_breakout",
    "chop",
)


@dataclass
class PathClassification:
    state: str
    confidence: float
    reasons: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)


# Indicator helpers (inline — Alpaca doesn't have indicators.py module)

def _rma(values: List[float], n: int) -> float:
    """Wilder's RMA (used for RSI/ATR smoothing)."""
    if len(values) < n:
        return sum(values) / max(1, len(values))
    val = sum(values[:n]) / n
    for v in values[n:]:
        val = (val * (n - 1) + v) / n
    return val


def _ema(closes: List[float], n: int) -> float:
    if not closes:
        return 0.0
    if len(closes) < n:
        return sum(closes) / len(closes)
    k = 2.0 / (n + 1.0)
    val = sum(closes[:n]) / n
    for p in closes[n:]:
        val = p * k + val * (1.0 - k)
    return val


def _rsi(closes: List[float], n: int = 14) -> float:
    if len(closes) < n + 1:
        return 50.0
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            avg_gain += d
        else:
            avg_loss -= d
    avg_gain /= n
    avg_loss /= n
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        g = d if d > 0 else 0.0
        l = (-d) if d < 0 else 0.0
        avg_gain = (avg_gain * (n - 1) + g) / n
        avg_loss = (avg_loss * (n - 1) + l) / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(highs, lows, closes, n: int = 14) -> float:
    if len(closes) < n + 1:
        return max(1e-9, closes[-1] * 0.003) if closes else 0.0
    trs = []
    for i in range(1, len(closes)):
        h = highs[i]; l = lows[i]; pc = closes[i - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < n:
        return max(1e-9, sum(trs) / len(trs))
    val = sum(trs[:n]) / n
    for tr in trs[n:]:
        val = (val * (n - 1) + tr) / n
    return max(1e-9, val)


def _rsi_series(closes: List[float], n: int = 14) -> List[float]:
    out = [50.0] * len(closes)
    if len(closes) < n + 1:
        return out
    avg_gain = 0.0; avg_loss = 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        if d >= 0: avg_gain += d
        else: avg_loss -= d
    avg_gain /= n; avg_loss /= n
    if avg_loss == 0:
        out[n] = 100.0
    else:
        out[n] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        g = d if d > 0 else 0.0
        l = (-d) if d < 0 else 0.0
        avg_gain = (avg_gain * (n - 1) + g) / n
        avg_loss = (avg_loss * (n - 1) + l) / n
        if avg_loss == 0:
            out[i] = 100.0
        else:
            out[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    return out


def _swing_pivots(highs: List[float], lows: List[float], fractal: int = 2) -> List[Tuple[int, str, float]]:
    pivots = []
    n = len(highs)
    for i in range(fractal, n - fractal):
        is_high = all(highs[i] >= highs[i + k] for k in range(-fractal, fractal + 1) if k != 0)
        is_low = all(lows[i] <= lows[i + k] for k in range(-fractal, fractal + 1) if k != 0)
        if is_high:
            pivots.append((i, "H", highs[i]))
        elif is_low:
            pivots.append((i, "L", lows[i]))
    return pivots


def _swing_count_last_n(pivots, n: int = 5) -> Dict[str, int]:
    counts = {"HH": 0, "HL": 0, "LL": 0, "LH": 0}
    if len(pivots) < 2:
        return counts
    ps = pivots[-n:]
    last_h = None; last_l = None
    for _, t, price in ps:
        if t == "H":
            if last_h is not None:
                counts["HH" if price > last_h else "LH"] += 1
            last_h = price
        else:
            if last_l is not None:
                counts["HL" if price > last_l else "LL"] += 1
            last_l = price
    return counts


def _rsi_curl(rsi_values, lookback: int = 3) -> int:
    if len(rsi_values) < lookback + 1:
        return 0
    diff = rsi_values[-1] - rsi_values[-1 - lookback]
    if diff > 1.0: return 1
    if diff < -1.0: return -1
    return 0


def _rsi_50_crosses(rsi_values, n: int = 24) -> int:
    if len(rsi_values) < n + 1:
        return 0
    crosses = 0
    series = rsi_values[-n - 1:]
    for i in range(1, len(series)):
        if (series[i - 1] < 50.0) != (series[i] < 50.0):
            crosses += 1
    return crosses


def _rsi_divergence(closes, rsi_values, pivots) -> Tuple[bool, bool]:
    bull_div = False; bear_div = False
    h_pivots = [p for p in pivots if p[1] == "H"][-3:]
    l_pivots = [p for p in pivots if p[1] == "L"][-3:]
    if len(h_pivots) >= 3:
        prices_up = h_pivots[-1][2] > h_pivots[-2][2] > h_pivots[-3][2]
        rsi_at = [rsi_values[idx] if idx < len(rsi_values) else 50.0 for idx, _, _ in h_pivots]
        if prices_up and rsi_at[-1] < rsi_at[-2]:
            bear_div = True
    if len(l_pivots) >= 3:
        prices_dn = l_pivots[-1][2] < l_pivots[-2][2] < l_pivots[-3][2]
        rsi_at = [rsi_values[idx] if idx < len(rsi_values) else 50.0 for idx, _, _ in l_pivots]
        if prices_dn and rsi_at[-1] > rsi_at[-2]:
            bull_div = True
    return bull_div, bear_div


def _three_push_up(rsi_values, pivots) -> bool:
    h = [p for p in pivots if p[1] == "H"][-3:]
    if len(h) < 3:
        return False
    p1, p2, p3 = h
    if not (p1[2] < p2[2] < p3[2]):
        return False
    if (p3[2] - p2[2]) < (p2[2] - p1[2]):
        return True
    r1 = rsi_values[p1[0]] if p1[0] < len(rsi_values) else 50.0
    r2 = rsi_values[p2[0]] if p2[0] < len(rsi_values) else 50.0
    r3 = rsi_values[p3[0]] if p3[0] < len(rsi_values) else 50.0
    return r3 < r2 < r1


def _trend_direction_closes(closes: List[float]) -> int:
    if len(closes) < 50:
        return 0
    e20 = _ema(closes, 20)
    e50 = _ema(closes, 50)
    if closes[-1] > e20 > e50:
        return 1
    if closes[-1] < e20 < e50:
        return -1
    return 0


# Main entry point — operates on 3 candle series (lists of dicts {ts, o, h, l, c, v})

def classify_path(symbol: str,
                  candles_5m: List[Dict],
                  candles_15m: List[Dict],
                  candles_1h: List[Dict],
                  prior_state: Optional[str] = None) -> PathClassification:
    """Classify path state for a stock symbol.

    Args:
        symbol: ticker e.g. 'AAPL'
        candles_5m, candles_15m, candles_1h: lists of dicts with keys o,h,l,c,v
        prior_state: state from prior cycle (drives breakout detection)
    """
    if len(candles_1h) < 50 or len(candles_5m) < 30 or len(candles_15m) < 20:
        return PathClassification(
            state="chop", confidence=0.3,
            reasons=["insufficient_data"],
            features={"symbol": symbol, "len_1h": len(candles_1h),
                      "len_5m": len(candles_5m), "len_15m": len(candles_15m)},
        )

    cl_1h = [c["c"] for c in candles_1h]
    h_1h = [c["h"] for c in candles_1h]
    l_1h = [c["l"] for c in candles_1h]
    cl_5m = [c["c"] for c in candles_5m]

    rsi_series_1h = _rsi_series(cl_1h, 14)
    pivots_1h = _swing_pivots(h_1h, l_1h, fractal=2)
    swings = _swing_count_last_n(pivots_1h, 5)
    bull_div, bear_div = _rsi_divergence(cl_1h, rsi_series_1h, pivots_1h)
    three_push = _three_push_up(rsi_series_1h, pivots_1h)

    rsi_now = rsi_series_1h[-1] if rsi_series_1h else 50.0
    rsi_curl = _rsi_curl(rsi_series_1h, 3)
    rsi_50_x = _rsi_50_crosses(rsi_series_1h, 24)

    e20_1h = _ema(cl_1h, 20)
    e50_1h = _ema(cl_1h, 50)
    close_1h = cl_1h[-1]

    atr_now = _atr(h_1h, l_1h, cl_1h, 14)
    atr_24h_ago = _atr(h_1h[:-24], l_1h[:-24], cl_1h[:-24], 14) if len(cl_1h) > 30 + 24 else atr_now
    atr_ratio = (atr_now / atr_24h_ago) if atr_24h_ago > 0 else 1.0
    atr_expanding = atr_ratio > 1.2
    atr_contracting = atr_ratio < 0.7

    # Volume advance/decline 5m × 12
    last_5m_12 = candles_5m[-12:] if len(candles_5m) >= 12 else candles_5m
    green_v = sum(c["v"] for c in last_5m_12 if c["c"] > c["o"])
    red_v = sum(c["v"] for c in last_5m_12 if c["c"] < c["o"])
    vol_adv_dec = (green_v / red_v) if red_v > 0 else 10.0
    vol_adv_dec = min(vol_adv_dec, 10.0)

    vol_breakout = 1.0
    if len(candles_5m) >= 21:
        avg_vol = sum(c["v"] for c in candles_5m[-21:-1]) / 20
        if avg_vol > 0:
            vol_breakout = candles_5m[-1]["v"] / avg_vol

    # 24h high/low from 5m (288 bars) — for stocks ~3.7 sessions of 5m
    sample_5m = candles_5m[-289:-1] if len(candles_5m) >= 289 else candles_5m[:-1]
    h_24h = max((c["h"] for c in sample_5m), default=0.0)
    l_24h = min((c["l"] for c in sample_5m), default=0.0)
    cl_5m_now = candles_5m[-1]["c"] if candles_5m else 0.0
    breakout_long_now = h_24h > 0 and cl_5m_now > h_24h and vol_breakout > 1.8
    breakout_short_now = l_24h > 0 and cl_5m_now < l_24h and vol_breakout > 1.8

    # Pullback to 5m EMA20
    pullback_5m = False
    near_5m = False
    if len(cl_5m) >= 20:
        e20_5m = _ema(cl_5m, 20)
        if e20_5m > 0:
            last_8 = candles_5m[-8:]
            pullback_5m = any(abs(c["c"] - e20_5m) / e20_5m < 0.003 for c in last_8)
            near_5m = abs(cl_5m_now - e20_5m) / e20_5m < 0.003

    # Three-push exhaustion / vol dry-up
    advances = [c for c in candles_1h[-30:] if c["c"] > c["o"]]
    vol_dry_up = False
    if len(advances) >= 12:
        last6 = advances[-6:]
        prior6 = advances[-12:-6]
        last_avg = sum(c["v"] for c in last6) / 6.0
        prior_avg = sum(c["v"] for c in prior6) / 6.0
        if prior_avg > 0:
            vol_dry_up = last_avg < 0.6 * prior_avg

    f = {
        "rsi_14_now": rsi_now,
        "rsi_14_curl": rsi_curl,
        "rsi_50_cross_count_24": rsi_50_x,
        "rsi_divergence_bull": bull_div,
        "rsi_divergence_bear": bear_div,
        "three_push_up_1h": three_push,
        "swing_HH": swings["HH"], "swing_HL": swings["HL"],
        "swing_LL": swings["LL"], "swing_LH": swings["LH"],
        "atr_14_1h": atr_now, "atr_ratio_24h": atr_ratio,
        "atr_expanding": atr_expanding, "atr_contracting": atr_contracting,
        "vol_advance_decline_5m_12": vol_adv_dec,
        "vol_breakout_ratio_5m": vol_breakout,
        "vol_dry_up_advance_1h_6": vol_dry_up,
        "prior_24h_high": h_24h, "prior_24h_low": l_24h,
        "breakout_long_now": breakout_long_now,
        "breakout_short_now": breakout_short_now,
        "pullback_5m_ema20": pullback_5m, "near_5m_ema20_now": near_5m,
        "ema20_1h": e20_1h, "ema50_1h": e50_1h, "close_1h": close_1h,
    }

    # Bullish exhaustion (any 2 of 4)
    be_score = 0; be_reasons = []
    if bear_div: be_score += 1; be_reasons.append("rsi_bearish_divergence")
    if three_push: be_score += 1; be_reasons.append("three_push_up_1h")
    if rsi_now > 75 and rsi_curl < 0: be_score += 1; be_reasons.append("rsi>75_curl_down")
    if vol_dry_up: be_score += 1; be_reasons.append("vol_dry_up_advance")
    if be_score >= 2:
        return PathClassification("bullish_exhaustion", min(1.0, 0.6 + 0.1 * (be_score - 2)), be_reasons, f)

    # Bearish exhaustion (any 2 of 4 — simplified for stocks)
    bex_score = 0; bex_reasons = []
    if bull_div: bex_score += 1; bex_reasons.append("rsi_bullish_divergence")
    if rsi_now < 25 and rsi_curl > 0: bex_score += 1; bex_reasons.append("rsi<25_curl_up")
    if bex_score >= 2:
        return PathClassification("bearish_exhaustion", 0.6, bex_reasons, f)

    # Failed breakout (recent breakout reclaimed)
    # Simplified: skip for stocks v1; classifier_log will refine

    # Breakout long
    if breakout_long_now and rsi_now < 75:
        if prior_state == "compression":
            return PathClassification("breakout_long", 0.8,
                                      ["prior_state=compression", "5m_break>24h_high", "vol_breakout>1.8"], f)
        return PathClassification("breakout_long", 0.55,
                                  ["5m_break>24h_high", "vol_breakout>1.8"], f)

    # Breakout short
    if breakout_short_now and rsi_now > 25:
        if prior_state == "compression":
            return PathClassification("breakout_short", 0.8,
                                      ["prior_state=compression", "5m_break<24h_low", "vol_breakout>1.8"], f)
        return PathClassification("breakout_short", 0.55,
                                  ["5m_break<24h_low", "vol_breakout>1.8"], f)

    # Bullish continuation (all required)
    bc1 = (swings["HH"] + swings["HL"]) >= 4
    bc2 = close_1h > e20_1h > e50_1h > 0
    bc3 = 45.0 <= rsi_now <= 70.0 and rsi_curl >= 0  # stock bands slightly wider
    bc4 = bool(pullback_5m or near_5m)
    bc5 = vol_adv_dec > 1.0
    bc_n = sum([bc1, bc2, bc3, bc4, bc5])
    if bc_n == 5:
        return PathClassification("bullish_continuation", min(1.0, 0.78 + 0.04 * (vol_adv_dec - 1.0)),
                                  ["1h_HH+HL>=4", "1h_close>EMA20>EMA50",
                                   "rsi_45-70_rising", "5m_pullback", "vol_adv>1"], f)
    if bc_n == 4:
        rs = []
        if bc1: rs.append("1h_HH+HL>=4")
        if bc2: rs.append("1h_close>EMA20>EMA50")
        if bc3: rs.append("rsi_45-70_rising")
        if bc4: rs.append("5m_pullback")
        if bc5: rs.append("vol_adv>1")
        return PathClassification("bullish_continuation", 0.62, rs, f)

    # Bearish continuation
    brc1 = (swings["LL"] + swings["LH"]) >= 4
    brc2 = e50_1h > 0 and close_1h < e20_1h < e50_1h
    brc3 = 30.0 <= rsi_now <= 55.0 and rsi_curl <= 0
    brc4 = vol_adv_dec < 1.0
    brc_n = sum([brc1, brc2, brc3, brc4])
    if brc_n == 4:
        return PathClassification("bearish_continuation", 0.78,
                                  ["1h_LL+LH>=4", "1h_close<EMA20<EMA50",
                                   "rsi_30-55_falling", "vol_dec>adv"], f)
    if brc_n == 3:
        rs = []
        if brc1: rs.append("1h_LL+LH>=4")
        if brc2: rs.append("1h_close<EMA20<EMA50")
        if brc3: rs.append("rsi_30-55_falling")
        if brc4: rs.append("vol_dec>adv")
        return PathClassification("bearish_continuation", 0.6, rs, f)

    # Compression
    range_12 = max((c["h"] for c in candles_1h[-12:]), default=0) - min((c["l"] for c in candles_1h[-12:]), default=0)
    range_ratio = (range_12 / atr_now) if atr_now > 0 else 0.0
    cp1 = atr_contracting
    cp2 = 0 < range_ratio < 1.5
    cp3 = 40.0 <= rsi_now <= 60.0
    cp_n = sum([cp1, cp2, cp3])
    if cp_n == 3:
        return PathClassification("compression", 0.8,
                                  ["atr_contracting", "range_12bar<1.5_atr", "rsi_pinned_40-60"], f)

    # Chop fallback
    swing_clean = max(swings["HH"] + swings["HL"], swings["LL"] + swings["LH"])
    rsi_pp = rsi_50_x >= 4
    no_struct = swing_clean < 3
    not_compress = not atr_contracting
    chop_n = sum([rsi_pp, no_struct, not_compress])
    chop_reasons = []
    if rsi_pp: chop_reasons.append("rsi_50cross>=4")
    if no_struct: chop_reasons.append("no_clean_swing")
    if not_compress: chop_reasons.append("atr_not_contracting")
    return PathClassification("chop", 0.3 + 0.15 * chop_n, chop_reasons, f)


def to_jsonable(c: PathClassification) -> Dict:
    feats = {}
    for k, v in c.features.items():
        if isinstance(v, bool):
            feats[k] = v
        elif isinstance(v, (int, float)):
            feats[k] = round(float(v), 6)
        else:
            feats[k] = v
    return {
        "state": c.state,
        "confidence": round(float(c.confidence), 3),
        "reasons": c.reasons,
        "features": feats,
    }
