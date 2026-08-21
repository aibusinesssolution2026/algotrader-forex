"""Forex broker layer.

PaperForexBroker: local simulated fills where cost = half-spread per side in
pips (how forex actually charges you) — no commissions on standard accounts.

OandaBroker: real OANDA v20 REST client. Triple-gated exactly like the
equities live broker:
  1. requires ALGO_TRADING_MODE=live + ALGO_LIVE_CONFIRM phrase,
  2. requires OANDA_API_TOKEN + OANDA_ACCOUNT_ID in the environment,
  3. targets the PRACTICE endpoint (fxpractice) unless OANDA_REAL_MONEY=yes.
A practice account at practice.oanda.com is free and is the correct way to
'deploy live' — real market data, simulated money.
"""
from __future__ import annotations

import logging
import os
import uuid

from ..config import resolve_trading_mode, LIVE_CONFIRM_PHRASE
from ..data.forex import PairSpec
from .broker import Broker, Fill

log = logging.getLogger(__name__)


class PaperForexBroker(Broker):
    def __init__(self, specs: dict[str, PairSpec]):
        from collections import deque
        self.specs = specs
        self.fills: "deque[Fill]" = deque(maxlen=10_000)

    def submit(self, symbol: str, side: int, qty: float,
               ref_price: float) -> Fill:
        spec = self.specs[symbol]
        half_spread = (spec.typical_spread_pips / 2) * spec.pip_size
        price = ref_price + side * half_spread     # cross the spread
        fill = Fill(order_id=uuid.uuid4().hex, symbol=symbol, side=side,
                    qty=qty, price=price, fee=0.0)
        self.fills.append(fill)
        log.info("FX PAPER FILL %s %+d x %.0f units @ %.5f "
                 "(spread cost %.1f pips)", symbol, side, qty, price,
                 spec.typical_spread_pips / 2)
        return fill


class OandaBroker(Broker):
    PRACTICE_URL = "https://api-fxpractice.oanda.com"
    LIVE_URL = "https://api-fxtrade.oanda.com"

    def __init__(self):
        if resolve_trading_mode() != "live":
            raise PermissionError(
                "OANDA broker blocked: set ALGO_TRADING_MODE=live and "
                f"ALGO_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}. For a practice "
                "account this is safe; it still uses simulated money unless "
                "OANDA_REAL_MONEY=yes is ALSO set.")
        self.token = os.environ.get("OANDA_API_TOKEN", "")
        self.account = os.environ.get("OANDA_ACCOUNT_ID", "")
        if not self.token or not self.account:
            raise PermissionError("Missing OANDA_API_TOKEN / OANDA_ACCOUNT_ID")
        real = os.environ.get("OANDA_REAL_MONEY", "") == "yes"
        self.base_url = self.LIVE_URL if real else self.PRACTICE_URL
        log.warning("OANDA broker -> %s (%s money)", self.base_url,
                    "REAL" if real else "practice")

    @staticmethod
    def _instrument(symbol: str) -> str:
        return f"{symbol[:3]}_{symbol[3:]}"        # EURUSD -> EUR_USD

    def submit(self, symbol: str, side: int, qty: float,
               ref_price: float) -> Fill:
        import requests
        units = int(side * qty)
        resp = requests.post(
            f"{self.base_url}/v3/accounts/{self.account}/orders",
            headers={"Authorization": f"Bearer {self.token}",
                     "Content-Type": "application/json"},
            json={"order": {"type": "MARKET",
                            "instrument": self._instrument(symbol),
                            "units": str(units),
                            "timeInForce": "FOK",
                            "positionFill": "DEFAULT"}},
            timeout=10)
        resp.raise_for_status()
        data = resp.json()
        tx = data.get("orderFillTransaction", {})
        return Fill(order_id=tx.get("id", uuid.uuid4().hex), symbol=symbol,
                    side=side, qty=abs(units),
                    price=float(tx.get("price", ref_price)), fee=0.0)
