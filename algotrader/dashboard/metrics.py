"""Live trading metrics computed from portfolio + executor state."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..execution.executor import Executor


@dataclass
class Metrics:
    equity: float
    win_loss_ratio: float | None
    win_count: int
    loss_count: int
    sharpe: float
    max_drawdown: float
    current_open_risk: float
    daily_drawdown: float
    n_trades: int
    halted: bool


class MetricsTracker:
    """Bounded memory: history capped via deque; max-drawdown uses a running
    peak so accuracy survives the cap in indefinitely long sessions."""

    def __init__(self, executor: Executor, max_history: int = 10_000):
        from collections import deque
        self.ex = executor
        self.equity_history: "deque[float]" = deque(maxlen=max_history)
        self._peak = 0.0
        self._worst_dd = 0.0

    def snapshot(self) -> Metrics:
        pf, prices = self.ex.portfolio, self.ex.prices
        eq = pf.equity(prices)
        self.equity_history.append(eq)
        self._peak = max(self._peak, eq)
        if self._peak > 0:
            self._worst_dd = min(self._worst_dd, eq / self._peak - 1)
        hist = np.asarray(self.equity_history)

        wins = sum(1 for p in self.ex.closed_pnls if p > 0)
        losses = sum(1 for p in self.ex.closed_pnls if p <= 0)
        # None when there's genuinely no data yet — 0.00 must mean a real
        # 0% win rate, never "we don't know." The dashboard renders these
        # differently (N/A vs 0.00) so the two cases are never confused.
        wl = (wins / losses if losses else float(wins)) if (wins or losses) \
            else None

        rets = np.diff(hist) / hist[:-1] if len(hist) > 2 else np.array([])
        sharpe = (float(np.sqrt(252) * rets.mean() / rets.std())
                  if rets.size > 1 and rets.std() > 0 else 0.0)
        mdd = self._worst_dd

        return Metrics(
            equity=eq,
            win_loss_ratio=wl,
            win_count=wins,
            loss_count=losses,
            sharpe=sharpe,
            max_drawdown=mdd,
            current_open_risk=pf.total_open_risk(),
            daily_drawdown=pf.daily_drawdown_pct(prices),
            n_trades=len(self.ex.closed_pnls),
            halted=self.ex.risk.halted,
        )
