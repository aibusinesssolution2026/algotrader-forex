"""Broker abstraction.

PaperBroker: fully local simulated fills with slippage + fees (the default).
AlpacaLiveBroker: stub that connects to Alpaca's endpoints — construction
raises unless the explicit live-mode environment double-opt-in is present,
and it defaults to Alpaca's *paper* endpoint even then.
"""
from __future__ import annotations

import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import BacktestConfig, resolve_trading_mode, LIVE_CONFIRM_PHRASE

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fill:
    order_id: str
    symbol: str
    side: int
    qty: float
    price: float
    fee: float


class Broker(ABC):
    @abstractmethod
    def submit(self, symbol: str, side: int, qty: float,
               ref_price: float) -> Fill: ...


class PaperBroker(Broker):
    def __init__(self, cfg: BacktestConfig | None = None,
                 max_fill_log: int = 10_000):
        from collections import deque
        self.cfg = cfg or BacktestConfig()
        self.fills: "deque[Fill]" = deque(maxlen=max_fill_log)

    def submit(self, symbol: str, side: int, qty: float,
               ref_price: float) -> Fill:
        slip = ref_price * self.cfg.slippage_bps / 10_000
        price = ref_price + side * slip
        fee = max(self.cfg.min_fee, qty * price * self.cfg.fee_bps / 10_000)
        fill = Fill(order_id=uuid.uuid4().hex, symbol=symbol, side=side,
                    qty=qty, price=price, fee=fee)
        self.fills.append(fill)
        log.info("PAPER FILL %s %+d x %.4f @ %.4f (fee %.2f)",
                 symbol, side, qty, price, fee)
        return fill


class AlpacaLiveBroker(Broker):
    """Guarded live broker. Instantiating this class without the explicit
    environment double-opt-in raises immediately. Even when permitted, the
    endpoint defaults to Alpaca's paper URL; the live URL requires a third
    env var. Keys are read from the environment — never hardcode them."""

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"

    def __init__(self):
        if resolve_trading_mode() != "live":
            raise PermissionError(
                "Live broker blocked: set ALGO_TRADING_MODE=live and "
                f"ALGO_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE} to enable. "
                "Paper trading is the default and recommended mode.")
        self.key = os.environ.get("ALPACA_API_KEY", "")
        self.secret = os.environ.get("ALPACA_API_SECRET", "")
        if not self.key or not self.secret:
            raise PermissionError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")
        use_real_money = os.environ.get("ALPACA_REAL_MONEY", "") == "yes"
        self.base_url = self.LIVE_URL if use_real_money else self.PAPER_URL
        log.warning("Live broker initialized against %s", self.base_url)

    def submit(self, symbol: str, side: int, qty: float,
               ref_price: float) -> Fill:
        import requests
        resp = requests.post(
            f"{self.base_url}/v2/orders",
            headers={"APCA-API-KEY-ID": self.key,
                     "APCA-API-SECRET-KEY": self.secret},
            json={"symbol": symbol, "qty": str(qty),
                  "side": "buy" if side == 1 else "sell",
                  "type": "market", "time_in_force": "day"},
            timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return Fill(order_id=data["id"], symbol=symbol, side=side,
                    qty=qty, price=ref_price, fee=0.0)
