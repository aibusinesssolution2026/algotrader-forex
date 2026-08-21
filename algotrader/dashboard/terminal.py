"""Real-time terminal dashboard using rich.Live. Zero web dependencies.
Usage:
    tracker = MetricsTracker(executor)
    run_terminal_dashboard(tracker, refresh_s=1.0)
"""
from __future__ import annotations

import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .metrics import MetricsTracker


def _sparkline(values, width: int = 40) -> str:
    values = list(values)          # accept deque or list
    if len(values) < 2:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(blocks[int((v - lo) / rng * (len(blocks) - 1))]
                   for v in vals)


def render(tracker: MetricsTracker) -> Panel:
    m = tracker.snapshot()
    t = Table.grid(padding=(0, 3))
    t.add_column(justify="right", style="bold cyan")
    t.add_column()
    t.add_row("Equity", f"${m.equity:,.2f}")
    t.add_row("Equity Curve", _sparkline(tracker.equity_history))
    t.add_row("Win/Loss Ratio", f"{m.win_loss_ratio:.2f}")
    t.add_row("Sharpe (ann.)", f"{m.sharpe:.2f}")
    t.add_row("Max Drawdown", f"{m.max_drawdown:.2%}")
    t.add_row("Daily Drawdown", f"{m.daily_drawdown:.2%}")
    t.add_row("Open Risk ($)", f"${m.current_open_risk:,.2f}")
    t.add_row("Closed Trades", str(m.n_trades))
    status = "[red bold]⛔ HALTED (circuit breaker)" if m.halted \
        else "[green]● RUNNING (paper)"
    t.add_row("Status", status)
    return Panel(t, title="AlgoTrader — Live Metrics", border_style="blue")


def run_terminal_dashboard(tracker: MetricsTracker,
                           refresh_s: float = 1.0) -> None:
    console = Console()
    with Live(render(tracker), console=console,
              refresh_per_second=max(1, int(1 / refresh_s))) as live:
        while tracker.ex.running or not tracker.ex.risk.halted:
            time.sleep(refresh_s)
            live.update(render(tracker))
            if tracker.ex.risk.halted:
                live.update(render(tracker))
                break
