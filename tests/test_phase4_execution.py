import time

import pytest

from algotrader.config import RiskConfig, BacktestConfig
from algotrader.execution.broker import PaperBroker, AlpacaLiveBroker
from algotrader.execution.executor import Executor, PriceEvent, SignalEvent
from algotrader.risk.engine import RiskEngine
from algotrader.state.portfolio import Portfolio
from algotrader.dashboard.metrics import MetricsTracker
from algotrader.dashboard.streamlit_app import write_metrics_file


def make_stack(cash=100_000.0):
    pf = Portfolio(cash=cash)
    risk = RiskEngine(RiskConfig(), pf)
    broker = PaperBroker(BacktestConfig())
    ex = Executor(risk, broker, pf)
    return ex, pf, risk, broker


def test_live_broker_blocked_without_env(monkeypatch):
    monkeypatch.delenv("ALGO_TRADING_MODE", raising=False)
    monkeypatch.delenv("ALGO_LIVE_CONFIRM", raising=False)
    with pytest.raises(PermissionError, match="Live broker blocked"):
        AlpacaLiveBroker()


def test_live_broker_blocked_without_keys(monkeypatch):
    monkeypatch.setenv("ALGO_TRADING_MODE", "live")
    monkeypatch.setenv("ALGO_LIVE_CONFIRM", "I_UNDERSTAND_THE_RISKS")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    with pytest.raises(PermissionError, match="Missing"):
        AlpacaLiveBroker()


def test_live_broker_defaults_to_paper_endpoint(monkeypatch):
    monkeypatch.setenv("ALGO_TRADING_MODE", "live")
    monkeypatch.setenv("ALGO_LIVE_CONFIRM", "I_UNDERSTAND_THE_RISKS")
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    monkeypatch.delenv("ALPACA_REAL_MONEY", raising=False)
    b = AlpacaLiveBroker()
    assert b.base_url == AlpacaLiveBroker.PAPER_URL


def test_signal_opens_position_with_stop():
    ex, pf, risk, broker = make_stack()
    ex.process(PriceEvent("AAPL", 150.0))
    ex.process(SignalEvent("AAPL", side=1, price=150.0, atr=3.0))
    assert "AAPL" in pf.positions
    pos = pf.positions["AAPL"]
    assert pos.stop_price == pytest.approx(150.0 - 2 * 3.0, rel=1e-3)
    assert len(broker.fills) == 1


def test_stop_loss_fires_on_price_breach():
    ex, pf, *_ = make_stack()
    ex.process(PriceEvent("AAPL", 150.0))
    ex.process(SignalEvent("AAPL", side=1, price=150.0, atr=3.0))
    stop = pf.positions["AAPL"].stop_price
    ex.process(PriceEvent("AAPL", stop - 0.01))
    assert "AAPL" not in pf.positions
    assert len(ex.closed_pnls) == 1
    assert ex.closed_pnls[0] < 0


def test_rejected_signal_never_reaches_broker():
    ex, pf, risk, broker = make_stack()
    ex.process(PriceEvent("AAPL", 150.0))
    ex.process(SignalEvent("AAPL", side=1, price=150.0, atr=0.0))  # bad ATR
    assert len(broker.fills) == 0
    assert pf.positions == {}
    assert risk.rejections


def test_event_loop_thread_processes_queue():
    ex, pf, *_ = make_stack()
    ex.start()
    ex.push(PriceEvent("MSFT", 300.0))
    ex.push(SignalEvent("MSFT", side=1, price=300.0, atr=5.0))
    deadline = time.monotonic() + 3
    while "MSFT" not in pf.positions and time.monotonic() < deadline:
        time.sleep(0.05)
    ex.stop()
    assert "MSFT" in pf.positions


def test_queue_bounded_drops_when_full():
    ex, *_ = make_stack()
    ex.events.maxsize = 2
    assert ex.push(PriceEvent("A", 1.0))
    assert ex.push(PriceEvent("A", 1.0))
    assert ex.push(PriceEvent("A", 1.0)) is False  # dropped, not blocked


def test_metrics_tracker_snapshot():
    ex, pf, *_ = make_stack()
    ex.process(PriceEvent("AAPL", 150.0))
    ex.process(SignalEvent("AAPL", side=1, price=150.0, atr=3.0))
    tracker = MetricsTracker(ex)
    m1 = tracker.snapshot()
    assert m1.current_open_risk > 0
    assert m1.n_trades == 0 and not m1.halted
    ex.process(PriceEvent("AAPL", 140.0))          # stop hit
    m2 = tracker.snapshot()
    assert m2.n_trades == 1
    assert m2.current_open_risk == 0.0
    assert m2.max_drawdown < 0


def test_metrics_file_atomic_write(tmp_path):
    p = tmp_path / "metrics.json"
    write_metrics_file({"equity": 1.0}, p)
    import json
    assert json.loads(p.read_text())["equity"] == 1.0
    assert not p.with_suffix(".tmp").exists()


def test_terminal_render_accepts_deque_history():
    from algotrader.dashboard.terminal import render, _sparkline
    ex, pf, *_ = make_stack()
    ex.process(PriceEvent("AAPL", 150.0))
    tracker = MetricsTracker(ex)
    for _ in range(5):
        tracker.snapshot()
    assert _sparkline(tracker.equity_history) is not None
    render(tracker)  # must not raise
