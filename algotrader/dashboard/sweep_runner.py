"""Runs the config sweep on demand from the web dashboard, in a background
thread so it doesn't block the trading loop or the HTTP server. Guarded
against overlapping runs — a second click while one is in progress is
ignored, not queued or stacked.
"""
from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger(__name__)


@dataclass
class SweepRunState:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    status: str = "idle"              # idle | running | done | error
    started_at: str | None = None
    finished_at: str | None = None
    results: dict = field(default_factory=dict)   # {pair: report_text}
    error: str | None = None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "results": dict(self.results),
                "error": self.error,
            }

    def start(self) -> bool:
        """Returns False if a sweep is already running (call ignored)."""
        with self._lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.started_at = datetime.now(timezone.utc).isoformat()
            self.finished_at = None
            self.error = None
            return True

    def finish(self, results: dict) -> None:
        with self._lock:
            self.status = "done"
            self.finished_at = datetime.now(timezone.utc).isoformat()
            self.results = results

    def fail(self, error: str) -> None:
        with self._lock:
            self.status = "error"
            self.finished_at = datetime.now(timezone.utc).isoformat()
            self.error = error


def run_sweep_background(pairs: list[str], offline: bool,
                         state: SweepRunState) -> None:
    """Runs synchronously on whatever thread calls it — caller is
    responsible for spawning the background thread."""
    from algotrader.data.forex import pair_spec, synthetic_fx, fetch_forex
    from algotrader.models.sweep import run_sweep, report

    try:
        results: dict[str, str] = {}
        for pair in pairs:
            spec = pair_spec(pair)
            if offline:
                base = synthetic_fx(pair, n=25_000, freq="min")
            else:
                data = fetch_forex([pair], period="5d", interval="1m")
                base = data.get(spec.symbol)
                if base is None or base.empty:
                    base = synthetic_fx(pair, n=25_000, freq="min")
            spread_bps = (spec.typical_spread_pips * spec.pip_size
                         / base["close"].mean()) * 10_000
            sweep_results = run_sweep(base, spread_bps)
            results[spec.symbol] = report(sweep_results)
            log.info("Sweep finished for %s", spec.symbol)
        state.finish(results)
    except Exception as exc:
        log.exception("Sweep failed")
        state.fail(f"{type(exc).__name__}: {exc}")


def trigger_sweep(pairs: list[str], offline: bool,
                  state: SweepRunState) -> bool:
    """Starts the sweep in a daemon thread. Returns False if one is
    already running (no-op in that case)."""
    if not state.start():
        return False
    t = threading.Thread(target=run_sweep_background,
                         args=(pairs, offline, state), daemon=True,
                         name="sweep-runner")
    t.start()
    return True
