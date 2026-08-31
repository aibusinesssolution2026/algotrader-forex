"""Event-style vectorless backtester: iterates bar-by-bar, applies slippage and
fees on every fill, and produces an equity curve plus trade log.

Strategy contract: a callable  (index, row, state) -> target signal in {-1,0,1}
Position sizing here is intentionally simple (fixed fraction); the hardened
Risk Engine (Phase 3) is layered on top in live/paper execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..config import BacktestConfig


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    side: int                      # +1 long, -1 short
    entry_price: float
    exit_price: float | None
    qty: float
    pnl: float = 0.0
    fees: float = 0.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        eq = self.equity_curve
        return float(eq.iloc[-1] / eq.iloc[0] - 1)

    @property
    def max_drawdown(self) -> float:
        eq = self.equity_curve
        dd = eq / eq.cummax() - 1
        return float(dd.min())

    @property
    def sharpe(self) -> float:
        rets = self.equity_curve.pct_change().dropna()
        if rets.std() == 0 or len(rets) < 2:
            return 0.0
        return float(np.sqrt(252) * rets.mean() / rets.std())

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.exit_price is not None]
        if not closed:
            return 0.0
        return sum(t.pnl > 0 for t in closed) / len(closed)


class Backtester:
    def __init__(self, cfg: BacktestConfig | None = None,
                 position_fraction: float = 0.1):
        self.cfg = cfg or BacktestConfig()
        self.position_fraction = position_fraction

    # -- cost model ------------------------------------------------------
    def _fill_price(self, price: float, side: int) -> float:
        """Adverse slippage: buys fill higher, sells fill lower."""
        slip = price * self.cfg.slippage_bps / 10_000
        return price + side * slip

    def _fee(self, notional: float) -> float:
        return max(self.cfg.min_fee, notional * self.cfg.fee_bps / 10_000)

    # -- main loop -------------------------------------------------------
    def run(self, df: pd.DataFrame,
            strategy: Callable[[pd.Timestamp, pd.Series, dict], int]
            ) -> BacktestResult:
        cash = self.cfg.initial_capital
        position: Trade | None = None
        trades: list[Trade] = []
        equity = []
        state: dict = {}

        for ts, row in df.iterrows():
            price = float(row["close"])
            signal = int(strategy(ts, row, state))

            # close position if signal flips or goes flat
            if position is not None and signal != position.side:
                fill = self._fill_price(price, -position.side)
                fee = self._fee(abs(fill * position.qty))
                gross = (fill - position.entry_price) * position.qty * position.side
                position.exit_time, position.exit_price = ts, fill
                position.fees += fee
                position.pnl = gross - position.fees
                cash += gross - fee
                trades.append(position)
                position = None

            # open new position
            if position is None and signal != 0:
                fill = self._fill_price(price, signal)
                notional = cash * self.position_fraction
                qty = notional / fill
                fee = self._fee(notional)
                cash -= fee
                position = Trade(entry_time=ts, exit_time=None, side=signal,
                                 entry_price=fill, exit_price=None,
                                 qty=qty, fees=fee)

            # mark-to-market equity
            unrealized = 0.0
            if position is not None:
                unrealized = ((price - position.entry_price)
                              * position.qty * position.side)
            equity.append(cash + unrealized)

        curve = pd.Series(equity, index=df.index, name="equity")
        return BacktestResult(equity_curve=curve, trades=trades)
