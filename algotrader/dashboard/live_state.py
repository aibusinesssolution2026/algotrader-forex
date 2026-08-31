"""Shared live state for the single-process web dashboard.

Design: the trading loop and the web server run as two threads inside ONE
process (not two Railway services), so there is no filesystem/volume
sharing problem to solve — state lives in memory and both threads read/
write the same object under a lock.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class LiveState:
    _lock: threading.RLock = field(default_factory=threading.RLock,
                                   repr=False)
    equity_history: "deque[tuple[str, float]]" = field(
        default_factory=lambda: deque(maxlen=1000))
    last_metrics: dict = field(default_factory=dict)
    recent_events: "deque[str]" = field(
        default_factory=lambda: deque(maxlen=100))
    open_positions: list = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def update(self, metrics: dict) -> None:
        with self._lock:
            self.last_metrics = metrics
            self.equity_history.append(
                (datetime.now(timezone.utc).isoformat(), metrics["equity"]))

    def log_event(self, text: str) -> None:
        with self._lock:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            self.recent_events.appendleft(f"[{ts}] {text}")

    def update_positions(self, positions: list[dict]) -> None:
        with self._lock:
            self.open_positions = positions

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "metrics": dict(self.last_metrics),
                "equity_history": list(self.equity_history),
                "events": list(self.recent_events),
                "open_positions": list(self.open_positions),
                "started_at": self.started_at,
            }
