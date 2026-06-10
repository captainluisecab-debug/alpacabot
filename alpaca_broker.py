"""
alpaca_broker.py — Order placement via Alpaca API.

Uses fractional shares so we can deploy exactly $80 regardless of stock price.
In PAPER mode: orders go to paper-api.alpaca.markets (safe).
In LIVE mode:  orders go to api.alpaca.markets (real money).
"""
from __future__ import annotations

import logging
from typing import Optional

from alpaca_settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, TRADE_MODE

log = logging.getLogger("alpaca_broker")


def _trading_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY,
                         paper=(TRADE_MODE == "PAPER"))


def get_account():
    """Return account info (cash, equity, buying power)."""
    try:
        return _trading_client().get_account()
    except Exception as exc:
        log.error("get_account failed: %s", exc)
        return None


def get_positions():
    """Return {symbol: position} for all open positions, or None on API ERROR.

    CRITICAL (D-034): an API error must return None, NOT {}. An empty dict means
    "account is genuinely flat"; returning {} on error made the engine reconcile
    nuke local positions on every transient failure, then re-adopt them via
    orphan_recovery next cycle — the phantom-position thrash. None lets the engine
    skip reconcile and preserve state this cycle.
    """
    try:
        client = _trading_client()
        positions = client.get_all_positions()
        return {p.symbol: p for p in positions}
    except Exception as exc:
        log.error("get_positions failed: %s", exc)
        return None


def get_recent_buy_fills(symbol: str, lookback_days: int = 60, limit: int = 500):
    """Return [(filled_qty, filled_avg_price, filled_at_epoch), ...] for recent CLOSED
    BUY orders, most-recent first. CORROBORATION ONLY for orphan adoption — never fatal.
    On any error returns [] (adoption falls back to Alpaca's authoritative avg_entry_price)."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import OrderSide, QueryOrderStatus
        from datetime import datetime, timezone, timedelta
        after = datetime.now(timezone.utc) - timedelta(days=int(lookback_days))
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, side=OrderSide.BUY,
                               symbols=[symbol], after=after, limit=int(limit), direction="desc")
        orders = _trading_client().get_orders(filter=req)
        out = []
        for o in orders:
            fq = float(getattr(o, "filled_qty", 0) or 0)
            fap = float(getattr(o, "filled_avg_price", 0) or 0)
            if fq > 0 and fap > 0:
                _fa = getattr(o, "filled_at", None)
                ts = int(_fa.timestamp()) if _fa else 0
                out.append((fq, fap, ts))
        return out
    except Exception as exc:
        log.warning("get_recent_buy_fills(%s) failed (corroboration only, non-fatal): %s", symbol, exc)
        return []


def _is_account_blocked_err(exc) -> bool:
    """True for Alpaca '40310000 / new orders are rejected by user request' —
    an ACCOUNT-LEVEL block (trade_suspended_by_user / trading_blocked), distinct
    from PDT (40310100). See D-034."""
    m = str(exc).lower()
    return ("40310000" in m) or ("rejected by user request" in m)


def buy_notional(symbol: str, usd_amount: float) -> Optional[dict]:
    """
    Buy $usd_amount worth of symbol using a notional (dollar-based) order.
    Alpaca supports fractional shares — no need to calculate share count.
    """
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    log.info("[%s] BUY $%.2f notional (%s)", TRADE_MODE, usd_amount, symbol)
    req = MarketOrderRequest(
        symbol=symbol,
        notional=round(usd_amount, 2),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    try:
        order = _trading_client().submit_order(req)
        log.info("[%s] Order submitted: %s %s $%.2f -> id=%s",
                 TRADE_MODE, order.side, symbol, usd_amount, order.id)
        return {"id": str(order.id), "symbol": symbol, "side": "BUY", "notional": usd_amount}
    except Exception as exc:
        if _is_account_blocked_err(exc):
            log.error("buy_notional(%s, %.2f) ACCOUNT-BLOCKED: %s", symbol, usd_amount, exc)
            return {"error": "account_blocked", "symbol": symbol}
        log.error("buy_notional(%s, %.2f) failed: %s", symbol, usd_amount, exc)
        return None


def sell_all(symbol: str) -> Optional[dict]:
    """Close the entire position in symbol.

    Returns:
      dict with id/symbol/side on success.
      dict with {"error": "pdt_block", "symbol": ...} on Alpaca PDT block (40310100).
      None on any other failure.
    """
    log.info("[%s] SELL ALL %s", TRADE_MODE, symbol)
    try:
        resp = _trading_client().close_position(symbol)
        log.info("[%s] Position closed: %s -> id=%s", TRADE_MODE, symbol, resp.id)
        return {"id": str(resp.id), "symbol": symbol, "side": "SELL"}
    except Exception as exc:
        # PDT detection — Alpaca returns 40310100 / "pattern day trading" for
        # sub-$25k accounts that exceed 3 day-trades in 5-day window. Surface
        # as tagged dict so engine can mark the symbol blocked-for-today
        # instead of retrying every cycle.
        _msg = str(exc).lower()
        if "40310100" in _msg or "pattern day trading" in _msg:
            log.warning("sell_all(%s) PDT-BLOCKED: %s", symbol, exc)
            return {"error": "pdt_block", "symbol": symbol}
        if _is_account_blocked_err(exc):
            log.error("sell_all(%s) ACCOUNT-BLOCKED: %s", symbol, exc)
            return {"error": "account_blocked", "symbol": symbol}
        log.error("sell_all(%s) failed: %s", symbol, exc)
        return None


def cancel_all_orders() -> None:
    """Cancel any open orders (cleanup utility)."""
    try:
        _trading_client().cancel_orders()
        log.info("All open orders cancelled")
    except Exception as exc:
        log.error("cancel_all_orders failed: %s", exc)
