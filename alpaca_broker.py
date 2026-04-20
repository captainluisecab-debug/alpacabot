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


def get_positions() -> dict:
    """Return {symbol: position} for all open positions."""
    try:
        client = _trading_client()
        positions = client.get_all_positions()
        return {p.symbol: p for p in positions}
    except Exception as exc:
        log.error("get_positions failed: %s", exc)
        return {}


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
        log.error("buy_notional(%s, %.2f) failed: %s", symbol, usd_amount, exc)
        return None


def sell_all(symbol: str) -> Optional[dict]:
    """Close the entire position in symbol."""
    log.info("[%s] SELL ALL %s", TRADE_MODE, symbol)
    try:
        resp = _trading_client().close_position(symbol)
        log.info("[%s] Position closed: %s -> id=%s", TRADE_MODE, symbol, resp.id)
        return {"id": str(resp.id), "symbol": symbol, "side": "SELL"}
    except Exception as exc:
        log.error("sell_all(%s) failed: %s", symbol, exc)
        return None


def cancel_all_orders() -> None:
    """Cancel any open orders (cleanup utility)."""
    try:
        _trading_client().cancel_orders()
        log.info("All open orders cancelled")
    except Exception as exc:
        log.error("cancel_all_orders failed: %s", exc)
