"""Portfolio state: single source of truth for cash, positions, and daily PnL.
Thread-safe: all mutation happens under a lock because the data stream,
risk engine, and executor may touch state from different threads.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date


@dataclass
class Position:
    symbol: str
    qty: float                 # signed: + long, - short
    entry_price: float
    stop_price: float

    @property
    def side(self) -> int:
        return 1 if self.qty > 0 else -1

    def open_risk(self) -> float:
        """Dollar amount lost if the stop is hit from entry."""
        return abs(self.entry_price - self.stop_price) * abs(self.qty)

    def unrealized(self, price: float) -> float:
        return (price - self.entry_price) * self.qty


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    day_start_equity: float = 0.0
    current_day: date | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock,
                                   repr=False, compare=False)

    def __post_init__(self):
        if self.day_start_equity == 0.0:
            self.day_start_equity = self.cash

    def equity(self, prices: dict[str, float]) -> float:
        with self._lock:
            mv = sum(p.unrealized(prices.get(p.symbol, p.entry_price))
                     for p in self.positions.values())
            return self.cash + mv

    def total_open_risk(self) -> float:
        with self._lock:
            return sum(p.open_risk() for p in self.positions.values())

    def roll_day(self, today: date, prices: dict[str, float]) -> None:
        """Reset the daily drawdown anchor at each new session."""
        with self._lock:
            if self.current_day != today:
                self.current_day = today
                self.day_start_equity = self.equity(prices)

    def daily_drawdown_pct(self, prices: dict[str, float]) -> float:
        with self._lock:
            if self.day_start_equity <= 0:
                return 0.0
            return self.equity(prices) / self.day_start_equity - 1

    def apply_fill(self, symbol: str, qty: float, price: float,
                   stop_price: float, fee: float = 0.0) -> None:
        with self._lock:
            self.cash -= fee
            if symbol in self.positions:
                pos = self.positions[symbol]
                new_qty = pos.qty + qty
                if abs(new_qty) < 1e-9:               # fully closed
                    self.cash += (price - pos.entry_price) * pos.qty
                    del self.positions[symbol]
                    return
                pos.qty = new_qty                      # partial adjust
            else:
                self.positions[symbol] = Position(symbol, qty, price,
                                                  stop_price)
