"""Forex executor: same event loop as the core Executor, but signals are
vetted through the ForexRiskEngine with pair specs and session gating."""
from __future__ import annotations

import logging

from ..data.forex import PairSpec
from ..risk.engine import OrderRequest, Verdict
from ..risk.forex import ForexRiskEngine
from .executor import Executor, SignalEvent

log = logging.getLogger(__name__)


class ForexExecutor(Executor):
    def __init__(self, risk: ForexRiskEngine, broker, portfolio,
                 specs: dict[str, PairSpec], max_queue: int = 10_000):
        super().__init__(risk, broker, portfolio, max_queue)
        self.specs = specs

    def _on_signal(self, ev: SignalEvent) -> None:
        if self.risk.halted:
            return
        spec = self.specs.get(ev.symbol)
        if spec is None:
            log.warning("No pair spec for %s; signal dropped", ev.symbol)
            return
        req = OrderRequest(symbol=ev.symbol, side=ev.side, price=ev.price,
                           atr=ev.atr)
        approval = self.risk.vet_order(
            req, spec, self.prices or {ev.symbol: ev.price}, ts=ev.ts)
        if approval.verdict is not Verdict.APPROVED:
            return
        self._execute(approval)
