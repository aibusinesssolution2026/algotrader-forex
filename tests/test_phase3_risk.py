import pytest

from algotrader.config import RiskConfig
from algotrader.state.portfolio import Portfolio, Position
from algotrader.risk.engine import (RiskEngine, PositionSizer, OrderRequest,
                                    Verdict)

CFG = RiskConfig()  # 1% risk, 10% notional cap, 3% daily DD, 2xATR stops


def make_engine(cash=100_000.0):
    pf = Portfolio(cash=cash)
    return RiskEngine(CFG, pf), pf


def test_sizing_risk_exactly_one_percent():
    sizer = PositionSizer(CFG)
    equity, price, atr = 100_000.0, 50.0, 5.0  # wide ATR: notional cap won't bind
    qty, stop = sizer.size(equity, price, atr, side=1)
    stop_dist = price - stop
    assert stop_dist == pytest.approx(CFG.atr_stop_multiplier * atr)
    loss_at_stop = qty * stop_dist
    assert loss_at_stop <= equity * CFG.max_risk_per_trade_pct + 1e-6
    assert loss_at_stop == pytest.approx(equity * CFG.max_risk_per_trade_pct,
                                         rel=1e-3)


def test_sizing_clamped_by_notional_cap():
    sizer = PositionSizer(CFG)
    # tiny ATR would imply a huge position; notional cap must bind
    qty, _ = sizer.size(100_000.0, 100.0, atr=0.01, side=1)
    assert qty * 100.0 <= 100_000.0 * CFG.max_position_pct + 1e-6


def test_sizing_short_stop_above_price():
    sizer = PositionSizer(CFG)
    _, stop = sizer.size(100_000.0, 50.0, 1.0, side=-1)
    assert stop > 50.0


def test_zero_atr_kills_order():
    eng, _ = make_engine()
    ap = eng.vet_order(OrderRequest("AAPL", 1, 150.0, atr=0.0),
                       {"AAPL": 150.0})
    assert ap.verdict is Verdict.REJECTED
    assert ap.qty == 0


def test_duplicate_position_rejected():
    eng, pf = make_engine()
    pf.positions["AAPL"] = Position("AAPL", 10, 150.0, 145.0)
    ap = eng.vet_order(OrderRequest("AAPL", 1, 150.0, atr=2.0),
                       {"AAPL": 150.0})
    assert ap.verdict is Verdict.REJECTED


def test_max_open_positions_enforced():
    eng, pf = make_engine()
    for i in range(CFG.max_open_positions):
        pf.positions[f"S{i}"] = Position(f"S{i}", 1, 10.0, 9.0)
    ap = eng.vet_order(OrderRequest("NEW", 1, 10.0, atr=0.5),
                       {"NEW": 10.0})
    assert ap.verdict is Verdict.REJECTED


def test_total_open_risk_cap():
    eng, pf = make_engine()
    # existing positions already carrying ~4.9% equity of open risk
    pf.positions["A"] = Position("A", 100, 100.0, 51.0)  # $4,900 risk
    ap = eng.vet_order(OrderRequest("B", 1, 50.0, atr=1.0), {"A": 100.0,
                                                             "B": 50.0})
    assert ap.verdict is Verdict.REJECTED
    assert "open risk" in ap.reason


def test_approved_order_token_single_use():
    eng, _ = make_engine()
    ap = eng.vet_order(OrderRequest("MSFT", 1, 300.0, atr=4.0),
                       {"MSFT": 300.0})
    assert ap.verdict is Verdict.APPROVED
    assert eng.consume_token(ap.token) is True
    assert eng.consume_token(ap.token) is False   # replay blocked


def test_forged_token_rejected():
    eng, _ = make_engine()
    assert eng.consume_token("deadbeef") is False


def test_circuit_breaker_trips_and_latches():
    eng, pf = make_engine(cash=100_000.0)
    pf.positions["XYZ"] = Position("XYZ", 1000, 100.0, 90.0)
    pf.day_start_equity = pf.equity({"XYZ": 100.0})
    # price collapses -> equity falls > 3% intraday
    crash = {"XYZ": 90.0}
    assert eng.check_circuit_breaker(crash) is True
    assert eng.halted
    # even after prices recover, breaker stays latched
    assert eng.check_circuit_breaker({"XYZ": 100.0}) is True
    ap = eng.vet_order(OrderRequest("AAPL", 1, 150.0, atr=2.0),
                       {"XYZ": 100.0, "AAPL": 150.0})
    assert ap.verdict is Verdict.HALTED


def test_liquidation_orders_flatten_everything():
    eng, pf = make_engine()
    pf.positions["A"] = Position("A", 200, 50.0, 45.0)
    pf.positions["B"] = Position("B", -40, 80.0, 88.0)
    pf.day_start_equity = pf.equity({"A": 50.0, "B": 80.0})
    eng.check_circuit_breaker({"A": 25.0, "B": 80.0})  # -5% equity: trips
    orders = eng.liquidation_orders({"A": 25.0, "B": 80.0})
    assert len(orders) == 2
    sides = {o.order.symbol: o.order.side for o in orders}
    assert sides["A"] == -1 and sides["B"] == 1        # opposite of position
    for o in orders:
        assert o.order.is_liquidation
        assert eng.consume_token(o.token)


def test_liquidation_refused_when_not_halted():
    eng, pf = make_engine()
    pf.positions["A"] = Position("A", 100, 50.0, 45.0)
    assert eng.liquidation_orders({"A": 50.0}) == []


def test_portfolio_daily_roll():
    from datetime import date
    pf = Portfolio(cash=100_000.0)
    pf.roll_day(date(2026, 7, 6), {})
    pf.cash = 98_000.0
    assert pf.daily_drawdown_pct({}) == pytest.approx(-0.02)
    pf.roll_day(date(2026, 7, 7), {})                  # new session resets
    assert pf.daily_drawdown_pct({}) == pytest.approx(0.0)
