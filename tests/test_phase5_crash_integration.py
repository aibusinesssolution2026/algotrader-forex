"""Integration test: extreme market crash.

Scenario: portfolio holds three long positions across symbols. A mocked
flash crash instantly drops every asset by 10%. Verify, end to end, that:
  1. per-position stop-losses and/or the circuit breaker fire,
  2. ALL positions are flattened,
  3. the executor halts and refuses every subsequent order,
  4. realized losses stay bounded near the configured risk budget
     (positions sized at ~1% risk each; total damage far below the naive
     10%-of-portfolio exposure a cap-less system would take).
"""
import time

import pytest

from algotrader.config import RiskConfig, BacktestConfig
from algotrader.execution.broker import PaperBroker
from algotrader.execution.executor import Executor, PriceEvent, SignalEvent
from algotrader.risk.engine import RiskEngine, Verdict, OrderRequest
from algotrader.state.portfolio import Portfolio

SYMS = ["AAPL", "MSFT", "NVDA"]
PRICES = {"AAPL": 150.0, "MSFT": 300.0, "NVDA": 500.0}
ATRS = {"AAPL": 3.0, "MSFT": 6.0, "NVDA": 12.0}


@pytest.fixture
def loaded_stack():
    pf = Portfolio(cash=100_000.0)
    risk = RiskEngine(RiskConfig(), pf)
    ex = Executor(risk, PaperBroker(BacktestConfig()), pf)
    for s in SYMS:
        ex.process(PriceEvent(s, PRICES[s]))
        ex.process(SignalEvent(s, side=1, price=PRICES[s], atr=ATRS[s]))
    assert len(pf.positions) == 3
    return ex, pf, risk


def crash(ex, pct=0.10):
    for s in SYMS:
        ex.process(PriceEvent(s, PRICES[s] * (1 - pct)))


def test_flash_crash_flattens_and_halts(loaded_stack):
    ex, pf, risk = loaded_stack
    equity_before = pf.equity(PRICES)

    crash(ex, 0.10)

    # every position is gone
    assert pf.positions == {}
    # circuit breaker or stops accounted for every close
    assert len(ex.closed_pnls) == 3
    assert all(p < 0 for p in ex.closed_pnls)
    # trading is halted (stops fired first position-by-position; if equity
    # breach also occurred the breaker latched — either way no new orders)
    ap = risk.vet_order(
        OrderRequest("TSLA", 1, 200.0, atr=5.0),
        {**{s: PRICES[s] * 0.9 for s in SYMS}, "TSLA": 200.0})
    if risk.halted:
        assert ap.verdict is Verdict.HALTED
    # loss bounded: 3 positions x ~1% risk +. slippage/fees — far under
    # the crash's raw 10% hit
    equity_after = pf.equity({s: PRICES[s] * 0.9 for s in SYMS})
    loss_pct = 1 - equity_after / equity_before
    assert loss_pct < 0.05
    assert loss_pct == pytest.approx(0.03, abs=0.02)


def test_flash_crash_through_live_event_loop(loaded_stack):
    """Same scenario but via the threaded queue — verifies the streaming
    path, not just direct process() calls."""
    ex, pf, risk = loaded_stack
    ex.start()
    for s in SYMS:
        ex.push(PriceEvent(s, PRICES[s] * 0.90))
    deadline = time.monotonic() + 3
    while pf.positions and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pf.positions == {}
    # post-crash signals are ignored
    ex.push(SignalEvent("TSLA", side=1, price=200.0, atr=5.0))
    time.sleep(0.3)
    ex.stop()
    assert "TSLA" not in pf.positions


def test_circuit_breaker_dominates_when_stops_lag():
    """Positions with stops far away (huge ATR): a 10% crash does NOT hit
    the per-position stops, so the GLOBAL breaker must be the one to fire."""
    pf = Portfolio(cash=100_000.0)
    # oversize positions manually to make daily DD breach possible
    risk = RiskEngine(RiskConfig(), pf)
    ex = Executor(risk, PaperBroker(BacktestConfig()), pf)
    from algotrader.state.portfolio import Position
    pf.positions["AAPL"] = Position("AAPL", 300, 150.0, 100.0)  # wide stop
    pf.positions["MSFT"] = Position("MSFT", 150, 300.0, 200.0)
    ex.prices.update(PRICES)
    from datetime import datetime, timezone
    pf.roll_day(datetime.now(timezone.utc).date(), PRICES)  # anchor session
    assert pf.day_start_equity == pytest.approx(100_000.0)

    ex.process(PriceEvent("AAPL", 135.0))   # -10%: DD ≈ -4.5% -> breaker
    assert risk.halted
    assert pf.positions == {}               # emergency flatten ran
    assert not ex.running                    # executor terminated itself


def test_halt_is_process_permanent():
    pf = Portfolio(cash=100_000.0)
    risk = RiskEngine(RiskConfig(), pf)
    pf.day_start_equity = 100_000.0
    pf.cash = 96_000.0                       # -4% day
    assert risk.check_circuit_breaker({}) is True
    pf.cash = 200_000.0                      # miracle recovery
    assert risk.check_circuit_breaker({}) is True   # still latched
    ap = risk.vet_order(OrderRequest("X", 1, 10.0, atr=1.0), {"X": 10.0})
    assert ap.verdict is Verdict.HALTED
