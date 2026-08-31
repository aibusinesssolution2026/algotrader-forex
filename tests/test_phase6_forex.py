from datetime import datetime, timezone

import pytest

from algotrader.config import RiskConfig
from algotrader.data.forex import (pair_spec, to_pips, synthetic_fx,
                                   market_open, rollover_window, yahoo_symbol)
from algotrader.data.preprocess import clean, engineer_features
from algotrader.execution.forex_broker import PaperForexBroker, OandaBroker
from algotrader.execution.forex_executor import ForexExecutor
from algotrader.execution.executor import PriceEvent, SignalEvent
from algotrader.risk.engine import RiskEngine, OrderRequest, Verdict
from algotrader.risk.forex import ForexRiskEngine, ForexRiskConfig
from algotrader.state.portfolio import Portfolio

MON_NOON = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)      # Monday
SAT = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)           # Saturday
SUN_LATE = datetime(2026, 7, 19, 22, 0, tzinfo=timezone.utc)      # Sun 22:00
FRI_LATE = datetime(2026, 7, 17, 21, 30, tzinfo=timezone.utc)     # Fri 21:30


def make_fx_stack(cash=100_000.0):
    pf = Portfolio(cash=cash)
    specs = {"EURUSD": pair_spec("EURUSD"), "USDJPY": pair_spec("USDJPY")}
    risk = ForexRiskEngine(RiskEngine(RiskConfig(), pf))
    broker = PaperForexBroker(specs)
    ex = ForexExecutor(risk, broker, pf, specs)
    return ex, pf, risk, broker


# ---------- pair conventions ----------

def test_pip_sizes():
    assert pair_spec("EURUSD").pip_size == 0.0001
    assert pair_spec("USDJPY").pip_size == 0.01
    assert pair_spec("eur/usd").symbol == "EURUSD"
    assert to_pips(pair_spec("EURUSD"), 0.0025) == pytest.approx(25.0)
    assert yahoo_symbol("EURUSD") == "EURUSD=X"


def test_synthetic_fx_reasonable():
    df = synthetic_fx("EURUSD", n=500)
    assert df["close"].between(0.8, 1.5).all()
    feats = engineer_features(clean(df))          # full pipeline compatible
    assert len(feats) > 400


# ---------- session calendar ----------

def test_market_hours():
    assert market_open(MON_NOON)
    assert not market_open(SAT)
    assert market_open(SUN_LATE)                  # opens Sunday 21:00 UTC
    assert not market_open(FRI_LATE)              # closed Friday 21:00 UTC
    assert rollover_window(datetime(2026, 7, 13, 22, 5,
                                    tzinfo=timezone.utc))


# ---------- forex risk engine ----------

def test_weekend_orders_killed():
    ex, pf, risk, broker = make_fx_stack()
    spec = pair_spec("EURUSD")
    ap = risk.vet_order(OrderRequest("EURUSD", 1, 1.10, atr=0.0015),
                        spec, {"EURUSD": 1.10}, ts=SAT)
    assert ap.verdict is Verdict.REJECTED
    assert "closed" in ap.reason


def test_rollover_entries_blocked():
    _, _, risk, _ = make_fx_stack()
    ap = risk.vet_order(OrderRequest("EURUSD", 1, 1.10, atr=0.0015),
                        pair_spec("EURUSD"), {"EURUSD": 1.10},
                        ts=datetime(2026, 7, 13, 22, 30, tzinfo=timezone.utc))
    assert ap.verdict is Verdict.REJECTED
    assert "rollover" in ap.reason


def test_tiny_stop_killed():
    _, _, risk, _ = make_fx_stack()
    # ATR so small the 2xATR stop is < 5 pips
    ap = risk.vet_order(OrderRequest("EURUSD", 1, 1.10, atr=0.0001),
                        pair_spec("EURUSD"), {"EURUSD": 1.10}, ts=MON_NOON)
    assert ap.verdict is Verdict.REJECTED
    assert "pips" in ap.reason


def test_leverage_cap_shrinks_position():
    pf = Portfolio(cash=10_000.0)
    # loose core caps so only the leverage cap binds
    core = RiskEngine(RiskConfig(max_position_pct=50.0,
                                 max_total_open_risk_pct=50.0,
                                 max_risk_per_trade_pct=0.02), pf)
    risk = ForexRiskEngine(core, ForexRiskConfig(max_leverage=5.0))
    ap = risk.vet_order(OrderRequest("EURUSD", 1, 1.10, atr=0.0004),
                        pair_spec("EURUSD"), {"EURUSD": 1.10}, ts=MON_NOON)
    assert ap.verdict is Verdict.APPROVED
    assert ap.qty > 0
    assert ap.qty * 1.10 <= 10_000.0 * 5.0 + 1.11   # notional <= 5x equity


def test_core_risk_still_applies():
    ex, pf, risk, _ = make_fx_stack()
    ap = risk.vet_order(OrderRequest("EURUSD", 5, 1.10, atr=0.0015),  # bad side
                        pair_spec("EURUSD"), {"EURUSD": 1.10}, ts=MON_NOON)
    assert ap.verdict is Verdict.REJECTED


# ---------- forex execution ----------

def test_fx_signal_opens_position_with_spread_cost():
    ex, pf, risk, broker = make_fx_stack()
    ex.process(PriceEvent("EURUSD", 1.1000))
    ex.process(SignalEvent("EURUSD", side=1, price=1.1000, atr=0.0015,
                           ts=MON_NOON))
    assert "EURUSD" in pf.positions
    fill = broker.fills[-1]
    spec = pair_spec("EURUSD")
    paid_spread_pips = to_pips(spec, fill.price - 1.1000)
    assert paid_spread_pips == pytest.approx(spec.typical_spread_pips / 2)


def test_fx_stop_loss_fires():
    ex, pf, *_ = make_fx_stack()
    ex.process(PriceEvent("EURUSD", 1.1000))
    ex.process(SignalEvent("EURUSD", side=1, price=1.1000, atr=0.0015,
                           ts=MON_NOON))
    stop = pf.positions["EURUSD"].stop_price
    ex.process(PriceEvent("EURUSD", stop - 0.0001))
    assert "EURUSD" not in pf.positions
    assert ex.closed_pnls and ex.closed_pnls[0] < 0


def test_fx_flash_crash_stops_bound_losses():
    """GBPUSD-style flash crash: -6% instant. First defense line (ATR stops)
    must flatten everything with losses bounded near the per-trade risk
    budget — WITHOUT needing the circuit breaker, which stays armed."""
    ex, pf, risk, _ = make_fx_stack()
    ex.process(PriceEvent("EURUSD", 1.1000))
    ex.process(PriceEvent("USDJPY", 150.00))
    for sym, px, atr in (("EURUSD", 1.1000, 0.0015), ("USDJPY", 150.0, 0.20)):
        ex.process(SignalEvent(sym, side=1, price=px, atr=atr, ts=MON_NOON))
    assert len(pf.positions) == 2
    ex.process(PriceEvent("EURUSD", 1.1000 * 0.94))
    ex.process(PriceEvent("USDJPY", 150.0 * 0.94))
    assert pf.positions == {}                       # all stops fired
    assert len(ex.closed_pnls) == 2
    crash_prices = {"EURUSD": 1.1000 * 0.94, "USDJPY": 150.0 * 0.94}
    dd = pf.daily_drawdown_pct(crash_prices)
    assert -0.03 < dd < 0                           # loss bounded under breaker
    assert not risk.halted                          # breaker armed, not needed


def test_fx_catastrophe_trips_breaker():
    """If positions are somehow oversized (e.g. legacy state), the breaker is
    the last line: it must flatten and latch."""
    ex, pf, risk, _ = make_fx_stack()
    from algotrader.state.portfolio import Position
    pf.positions["EURUSD"] = Position("EURUSD", 400_000, 1.1000, 0.9000)
    ex.prices.update({"EURUSD": 1.1000})
    pf.roll_day(MON_NOON.date(), {"EURUSD": 1.1000})
    ex.process(PriceEvent("EURUSD", 1.1000 * 0.99, ts=MON_NOON))  # -1% x 4.4x notional
    assert risk.halted
    assert pf.positions == {}
    ap = risk.vet_order(OrderRequest("USDJPY", 1, 150.0, atr=0.2),
                        pair_spec("USDJPY"), {"EURUSD": 1.089,
                                              "USDJPY": 150.0}, ts=MON_NOON)
    assert ap.verdict is Verdict.HALTED


# ---------- broker guards ----------

def test_oanda_blocked_without_env(monkeypatch):
    monkeypatch.delenv("ALGO_TRADING_MODE", raising=False)
    monkeypatch.delenv("ALGO_LIVE_CONFIRM", raising=False)
    with pytest.raises(PermissionError, match="blocked"):
        OandaBroker()


def test_oanda_defaults_to_practice(monkeypatch):
    monkeypatch.setenv("ALGO_TRADING_MODE", "live")
    monkeypatch.setenv("ALGO_LIVE_CONFIRM", "I_UNDERSTAND_THE_RISKS")
    monkeypatch.setenv("OANDA_API_TOKEN", "t")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "a")
    monkeypatch.delenv("OANDA_REAL_MONEY", raising=False)
    b = OandaBroker()
    assert b.base_url == OandaBroker.PRACTICE_URL
    assert OandaBroker._instrument("EURUSD") == "EUR_USD"
