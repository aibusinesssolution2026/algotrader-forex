"""Forex-specific risk layer, wrapping the core RiskEngine.

Adds what equities didn't need:
  - unit-based sizing (forex trades in currency units, not shares),
  - a HARD leverage cap (retail forex leverage is the account-killer),
  - session gating: no new entries when the market is closed or during the
    daily rollover spread blow-out.

The core engine's caps (per-trade risk, open-risk, breaker) still apply —
this wrapper only ever makes orders SMALLER or kills them, never bigger.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from ..config import RiskConfig
from ..data.forex import PairSpec, market_open, rollover_window
from ..risk.engine import RiskEngine, OrderRequest, RiskApproval, Verdict
from ..state.portfolio import Portfolio


@dataclass(frozen=True)
class ForexRiskConfig:
    max_leverage: float = 5.0        # notional / equity, deliberately far
                                     # below the 30-50x brokers offer
    min_stop_pips: float = 5.0       # refuse stops tighter than spread noise
    block_rollover_entries: bool = True


class ForexRiskEngine:
    def __init__(self, core: RiskEngine, fx_cfg: ForexRiskConfig | None = None):
        self.core = core
        self.fx = fx_cfg or ForexRiskConfig()
        # forex is margin-traded: cash check = notional / leverage
        self.core.margin_divisor = self.fx.max_leverage

    @property
    def halted(self) -> bool:
        return self.core.halted

    def vet_order(self, req: OrderRequest, spec: PairSpec,
                  prices: dict[str, float],
                  ts: datetime | None = None) -> RiskApproval:
        ts = ts or datetime.now(timezone.utc)

        def kill(reason: str) -> RiskApproval:
            self.core.rejections.append(f"{req.symbol}: {reason}")
            return RiskApproval(token="", order=req, qty=0.0, stop_price=0.0,
                                verdict=Verdict.REJECTED, reason=reason)

        # session gates (before touching the core engine)
        if not market_open(ts):
            return kill("forex market closed (weekend gap)")
        if self.fx.block_rollover_entries and rollover_window(ts):
            return kill("rollover window: spreads unreliable")

        # stop distance sanity in pips
        stop_dist = self.core.cfg.atr_stop_multiplier * req.atr
        if stop_dist / spec.pip_size < self.fx.min_stop_pips:
            return kill(f"stop < {self.fx.min_stop_pips} pips: inside spread noise")

        approval = self.core.vet_order(req, prices)
        if approval.verdict is not Verdict.APPROVED:
            return approval

        # leverage cap: shrink units if notional exceeds equity * max_leverage
        equity = self.core.portfolio.equity(prices)
        max_notional = equity * self.fx.max_leverage
        notional = approval.qty * req.price
        if notional > max_notional:
            capped_qty = math.floor(max_notional / req.price)
            # re-issue a smaller approval under the same token
            approval = RiskApproval(token=approval.token, order=req,
                                    qty=float(capped_qty),
                                    stop_price=approval.stop_price,
                                    verdict=Verdict.APPROVED,
                                    reason="leverage-capped")
        return approval

    # passthroughs so the executor works unchanged
    def consume_token(self, token: str) -> bool:
        return self.core.consume_token(token)

    def check_circuit_breaker(self, prices) -> bool:
        return self.core.check_circuit_breaker(prices)

    def liquidation_orders(self, prices):
        return self.core.liquidation_orders(prices)

    def approve_close(self, symbol, price, reason):
        return self.core.approve_close(symbol, price, reason)

    @property
    def portfolio(self) -> Portfolio:
        return self.core.portfolio

    @property
    def cfg(self) -> RiskConfig:
        return self.core.cfg

    @property
    def rejections(self):
        return self.core.rejections
