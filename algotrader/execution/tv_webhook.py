"""TradingView webhook receiver.

TradingView alert -> POST /webhook -> validated -> ForexRiskEngine -> paper
execution. TradingView signals are treated as just another strategy source:
every alert passes the SAME risk gates as the swarm (sizing, leverage,
session, breaker). Nothing about coming from TradingView earns an alert any
trust.

Security model (TradingView can only send a static JSON body — no headers):
  - shared secret INSIDE the JSON payload, compared with hmac.compare_digest
    (constant-time; set TV_WEBHOOK_SECRET in the environment)
  - strict payload schema; unknown symbols rejected
  - per-symbol cooldown to neutralize alert storms / duplicate deliveries
  - bounded in-memory alert log

Expected TradingView alert message body:
    {"secret": "<TV_WEBHOOK_SECRET>",
     "symbol": "EURUSD",
     "action": "buy" | "sell" | "close",
     "price": {{close}},
     "atr": <optional, price units>}

Strategy comparison: alerts are tagged source="tradingview" in the shared
trade log so their paper performance can be measured against the swarm's.
"""
from __future__ import annotations

import hmac
import logging
import os
import time
from collections import deque
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..data.forex import pair_spec
from ..execution.executor import PriceEvent, SignalEvent
from ..execution.forex_executor import ForexExecutor

log = logging.getLogger(__name__)

COOLDOWN_SECONDS = 60.0


class TVAlert(BaseModel):
    secret: str
    symbol: str = Field(min_length=6, max_length=10)
    action: Literal["buy", "sell", "close"]
    price: float = Field(gt=0)
    atr: float | None = Field(default=None, gt=0)


class WebhookState:
    def __init__(self, executor: ForexExecutor, secret: str):
        if not secret or len(secret) < 16:
            raise ValueError(
                "TV_WEBHOOK_SECRET must be set and >= 16 chars. Generate one:"
                "  python -c \"import secrets; print(secrets.token_hex(24))\"")
        self.executor = executor
        self.secret = secret
        self.last_alert: dict[str, float] = {}
        self.alert_log: deque = deque(maxlen=5_000)


def create_app(state: WebhookState) -> FastAPI:
    app = FastAPI(title="AlgoTrader TradingView Bridge", docs_url=None,
                  redoc_url=None)

    @app.post("/webhook")
    def webhook(alert: TVAlert):
        # constant-time secret check
        if not hmac.compare_digest(alert.secret, state.secret):
            log.warning("Webhook: bad secret from alert on %s", alert.symbol)
            raise HTTPException(status_code=403, detail="forbidden")

        sym = alert.symbol.upper().replace("/", "").replace("=X", "")
        if sym not in state.executor.specs:
            raise HTTPException(status_code=400,
                                detail=f"unknown symbol {sym}")

        now = time.monotonic()
        if alert.action != "close":
            if now - state.last_alert.get(sym, -1e9) < COOLDOWN_SECONDS:
                state.alert_log.append((sym, alert.action, "cooldown"))
                return {"status": "ignored", "reason": "cooldown"}
            state.last_alert[sym] = now

        ex = state.executor
        ex.process(PriceEvent(sym, alert.price))

        if ex.risk.halted:
            state.alert_log.append((sym, alert.action, "halted"))
            return {"status": "rejected", "reason": "circuit breaker halted"}

        if alert.action == "close":
            in_book = sym in ex.portfolio.positions
            if in_book:
                ex._close_position(sym, alert.price, reason="tv-close")
            state.alert_log.append((sym, "close",
                                    "closed" if in_book else "no position"))
            return {"status": "ok" if in_book else "ignored",
                    "reason": "closed" if in_book else "no open position"}

        atr = alert.atr or _default_atr(sym, alert.price)
        side = 1 if alert.action == "buy" else -1
        before = len(ex.portfolio.positions)
        ex.process(SignalEvent(sym, side, alert.price, atr))
        opened = len(ex.portfolio.positions) > before
        reason = "filled" if opened else \
            (list(ex.risk.rejections)[-1] if ex.risk.rejections
             else "rejected by risk engine")
        state.alert_log.append((sym, alert.action,
                                "filled" if opened else reason))
        return {"status": "ok" if opened else "rejected", "reason": reason,
                "source": "tradingview"}

    @app.get("/health")
    def health():
        pf = state.executor.portfolio
        return {"halted": state.executor.risk.halted,
                "open_positions": list(pf.positions),
                "alerts_seen": len(state.alert_log)}

    return app


def _default_atr(sym: str, price: float) -> float:
    """If the alert omits ATR, assume a conservative 0.3% of price — wider
    stop, smaller position. Never guess a tight stop."""
    spec = pair_spec(sym)
    return max(price * 0.003, 10 * spec.pip_size)
