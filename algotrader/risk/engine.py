"""The hardened Risk Engine. Every order MUST pass through `vet_order`.
There is no code path from strategy to broker that bypasses this layer —
the executor refuses orders lacking a valid RiskApproval token.

Rules enforced (all from the frozen RiskConfig — not mutable at runtime):
  1. Per-trade risk cap: position sized so a stop-out loses at most
     max_risk_per_trade_pct of current equity (ATR-based stop distance).
  2. Position notional cap: max_position_pct of equity.
  3. Portfolio-wide open-risk cap.
  4. Max concurrent positions.
  5. GLOBAL CIRCUIT BREAKER: daily drawdown beyond max_daily_drawdown_pct
     trips a latching halt — all positions are flattened and no new order
     is ever approved again in this process.
"""
from __future__ import annotations

import logging
import math
import threading
import uuid
from dataclasses import dataclass
from enum import Enum

from ..config import RiskConfig
from ..state.portfolio import Portfolio

log = logging.getLogger(__name__)


class Verdict(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    HALTED = "halted"


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: int                  # +1 buy/long, -1 sell/short
    price: float               # current reference price
    atr: float                 # current ATR for stop placement
    is_liquidation: bool = False   # circuit-breaker flatten orders


@dataclass(frozen=True)
class RiskApproval:
    """Unforgeable-by-convention token: only the RiskEngine constructs these,
    and the executor validates the token id against the engine's registry."""
    token: str
    order: OrderRequest
    qty: float
    stop_price: float
    verdict: Verdict
    reason: str = ""


class PositionSizer:
    """ATR-based sizing:  qty = (equity * risk_pct) / (atr * stop_multiplier).
    The stop sits `stop_multiplier * ATR` away, so hitting it loses exactly
    the risk budget (before slippage)."""

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def size(self, equity: float, price: float, atr: float,
             side: int) -> tuple[float, float]:
        if atr <= 0 or price <= 0 or equity <= 0:
            return 0.0, 0.0
        stop_dist = self.cfg.atr_stop_multiplier * atr
        risk_budget = equity * self.cfg.max_risk_per_trade_pct
        qty = risk_budget / stop_dist
        # clamp by notional cap
        max_qty_notional = (equity * self.cfg.max_position_pct) / price
        qty = min(qty, max_qty_notional)
        qty = math.floor(qty * 1e4) / 1e4   # truncate, never round up risk
        stop_price = price - side * stop_dist
        return qty, stop_price


class RiskEngine:
    def __init__(self, cfg: RiskConfig, portfolio: Portfolio,
                 margin_divisor: float = 1.0):
        """margin_divisor=1.0 -> full cash required (equities, cash account).
        margin_divisor=N    -> only notional/N cash required (margin FX)."""
        self.cfg = cfg
        self.portfolio = portfolio
        self.margin_divisor = max(1.0, margin_divisor)
        self.sizer = PositionSizer(cfg)
        self._halted = threading.Event()
        self._valid_tokens: set[str] = set()
        self._lock = threading.Lock()
        from collections import deque
        self.rejections: "deque[str]" = deque(maxlen=5_000)

    # -- circuit breaker --------------------------------------------------
    @property
    def halted(self) -> bool:
        return self._halted.is_set()

    def check_circuit_breaker(self, prices: dict[str, float]) -> bool:
        """Returns True if trading is (now) halted. Latching: once tripped,
        it never resets within the process lifetime."""
        if self._halted.is_set():
            return True
        dd = self.portfolio.daily_drawdown_pct(prices)
        if dd <= -self.cfg.max_daily_drawdown_pct:
            self._halted.set()
            log.critical("CIRCUIT BREAKER TRIPPED: daily drawdown %.2f%% "
                         "breached limit %.2f%%. Halting all trading.",
                         dd * 100, -self.cfg.max_daily_drawdown_pct * 100)
            return True
        return False

    def liquidation_orders(self, prices: dict[str, float]) -> list[RiskApproval]:
        """Emergency flatten: pre-approved closing orders for every open
        position. Only callable once halted."""
        if not self.halted:
            return []
        approvals = []
        with self._lock:
            for pos in list(self.portfolio.positions.values()):
                req = OrderRequest(symbol=pos.symbol, side=-pos.side,
                                   price=prices.get(pos.symbol,
                                                    pos.entry_price),
                                   atr=0.0, is_liquidation=True)
                token = uuid.uuid4().hex
                self._valid_tokens.add(token)
                approvals.append(RiskApproval(
                    token=token, order=req, qty=abs(pos.qty),
                    stop_price=0.0, verdict=Verdict.APPROVED,
                    reason="circuit-breaker liquidation"))
        return approvals

    # -- order vetting ----------------------------------------------------
    def vet_order(self, req: OrderRequest,
                  prices: dict[str, float]) -> RiskApproval:
        def reject(reason: str, verdict: Verdict = Verdict.REJECTED):
            self.rejections.append(f"{req.symbol}: {reason}")
            log.warning("ORDER KILLED [%s]: %s", req.symbol, reason)
            return RiskApproval(token="", order=req, qty=0.0, stop_price=0.0,
                                verdict=verdict, reason=reason)

        # 0. circuit breaker gates everything except liquidations
        if self.check_circuit_breaker(prices) and not req.is_liquidation:
            return reject("global circuit breaker active", Verdict.HALTED)

        if req.side not in (-1, 1):
            return reject("invalid side")
        if req.symbol in self.portfolio.positions and not req.is_liquidation:
            return reject("position already open for symbol")
        if len(self.portfolio.positions) >= self.cfg.max_open_positions:
            return reject("max open positions reached")

        equity = self.portfolio.equity(prices)
        qty, stop_price = self.sizer.size(equity, req.price, req.atr, req.side)
        if qty <= 0:
            return reject("sized to zero (bad ATR/price/equity)")

        # affordability clamp: full notional (cash acct, divisor=1) or
        # margin requirement (divisor=leverage). Shrink, don't reject.
        affordable_qty = max(self.portfolio.cash, 0) * self.margin_divisor             / req.price
        if qty > affordable_qty:
            qty = math.floor(affordable_qty * 1e4) / 1e4
        if qty <= 0:
            return reject("insufficient cash/margin for any size")

        # portfolio-wide open risk cap
        new_risk = abs(req.price - stop_price) * qty
        if (self.portfolio.total_open_risk() + new_risk
                > equity * self.cfg.max_total_open_risk_pct):
            return reject("total open risk cap exceeded")

        with self._lock:
            token = uuid.uuid4().hex
            self._valid_tokens.add(token)
        return RiskApproval(token=token, order=req, qty=qty,
                            stop_price=stop_price, verdict=Verdict.APPROVED)

    def approve_close(self, symbol: str, price: float,
                      reason: str) -> RiskApproval | None:
        """Public API for risk-reducing closes (stop-loss / manual exit).
        Closing risk is always allowed, even when halted."""
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return None
        req = OrderRequest(symbol=symbol, side=-pos.side, price=price,
                           atr=0.0, is_liquidation=True)
        with self._lock:
            token = uuid.uuid4().hex
            self._valid_tokens.add(token)
        return RiskApproval(token=token, order=req, qty=abs(pos.qty),
                            stop_price=0.0, verdict=Verdict.APPROVED,
                            reason=reason)

    def consume_token(self, token: str) -> bool:
        """One-time-use validation performed by the executor."""
        with self._lock:
            if token in self._valid_tokens:
                self._valid_tokens.remove(token)
                return True
        return False
