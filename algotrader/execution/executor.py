"""Event-driven execution layer.

Events flow through a bounded queue:
    PriceEvent  -> updates marks, checks stops + circuit breaker
    SignalEvent -> risk-vetted, then routed to broker

The executor is the ONLY component that talks to the broker, and it refuses
any order that does not carry a one-time RiskApproval token issued by the
RiskEngine. Bounded queue prevents unbounded memory growth in long-running
streaming loops; stale events are dropped with a warning.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..execution.broker import Broker
from ..risk.engine import RiskEngine, OrderRequest, Verdict, RiskApproval
from ..state.portfolio import Portfolio

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriceEvent:
    symbol: str
    price: float
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class SignalEvent:
    symbol: str
    side: int
    price: float
    atr: float
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Executor:
    def __init__(self, risk: RiskEngine, broker: Broker, portfolio: Portfolio,
                 max_queue: int = 10_000):
        self.risk = risk
        self.broker = broker
        self.portfolio = portfolio
        self.events: queue.Queue = queue.Queue(maxsize=max_queue)
        self.prices: dict[str, float] = {}
        self.closed_pnls: list[float] = []
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

    # -- public API --------------------------------------------------------
    def push(self, event) -> bool:
        try:
            self.events.put_nowait(event)
            return True
        except queue.Full:
            log.warning("Event queue full; dropping %s", type(event).__name__)
            return False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="executor")
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() \
            and not self._stop_flag.is_set()

    # -- event loop ----------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_flag.is_set():
            try:
                event = self.events.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self.process(event)
            except Exception:
                log.exception("Error processing %s", event)
            finally:
                self.events.task_done()

    def process(self, event) -> None:
        if isinstance(event, PriceEvent):
            self._on_price(event)
        elif isinstance(event, SignalEvent):
            self._on_signal(event)

    # -- handlers -----------------------------------------------------------
    def _on_price(self, ev: PriceEvent) -> None:
        self.prices[ev.symbol] = ev.price
        self.portfolio.roll_day(ev.ts.date(), self.prices)

        # 1. circuit breaker check on every tick — flatten + halt if tripped
        if self.risk.check_circuit_breaker(self.prices):
            self._emergency_flatten()
            return

        # 2. per-position stop-loss monitoring
        pos = self.portfolio.positions.get(ev.symbol)
        if pos is not None:
            stop_hit = (pos.side == 1 and ev.price <= pos.stop_price) or \
                       (pos.side == -1 and ev.price >= pos.stop_price)
            if stop_hit:
                log.warning("STOP HIT %s @ %.4f (stop %.4f)",
                            ev.symbol, ev.price, pos.stop_price)
                self._close_position(ev.symbol, ev.price,
                                     reason="stop-loss")

    def _on_signal(self, ev: SignalEvent) -> None:
        if self.risk.halted:
            return
        req = OrderRequest(symbol=ev.symbol, side=ev.side, price=ev.price,
                           atr=ev.atr)
        approval = self.risk.vet_order(req, self.prices or
                                       {ev.symbol: ev.price})
        if approval.verdict is not Verdict.APPROVED:
            return
        self._execute(approval)

    # -- order routing (token-gated) ------------------------------------------
    def _execute(self, approval: RiskApproval) -> None:
        if not self.risk.consume_token(approval.token):
            log.error("SECURITY: rejected order with invalid/reused token")
            return
        o = approval.order
        fill = self.broker.submit(o.symbol, o.side, approval.qty, o.price)
        signed_qty = fill.side * fill.qty
        self.portfolio.apply_fill(o.symbol, signed_qty, fill.price,
                                  approval.stop_price, fill.fee)

    def _close_position(self, symbol: str, price: float, reason: str) -> None:
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return
        approval = self.risk.approve_close(symbol, price, reason)
        if approval is None:
            return
        pnl_before = pos.unrealized(price)
        self._execute(approval)
        self.closed_pnls.append(pnl_before)

    def _emergency_flatten(self) -> None:
        for approval in self.risk.liquidation_orders(self.prices):
            pos = self.portfolio.positions.get(approval.order.symbol)
            if pos is not None:
                self.closed_pnls.append(
                    pos.unrealized(approval.order.price))
            self._execute(approval)
        log.critical("Emergency flatten complete. Executor halting.")
        self._stop_flag.set()
