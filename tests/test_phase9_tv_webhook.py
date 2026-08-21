import pytest
from fastapi.testclient import TestClient

from algotrader.config import RiskConfig
from algotrader.data.forex import pair_spec
from algotrader.execution.forex_broker import PaperForexBroker
from algotrader.execution.forex_executor import ForexExecutor
from algotrader.execution.tv_webhook import (create_app, WebhookState,
                                             COOLDOWN_SECONDS)
from algotrader.risk.engine import RiskEngine
from algotrader.risk.forex import ForexRiskEngine
from algotrader.state.portfolio import Portfolio

SECRET = "a" * 32


@pytest.fixture
def client(monkeypatch):
    # freeze session gates open: patch market_open used by the risk wrapper
    monkeypatch.setattr("algotrader.risk.forex.market_open", lambda ts: True)
    monkeypatch.setattr("algotrader.risk.forex.rollover_window",
                        lambda ts: False)
    pf = Portfolio(cash=100_000.0)
    specs = {"EURUSD": pair_spec("EURUSD")}
    risk = ForexRiskEngine(RiskEngine(RiskConfig(), pf))
    ex = ForexExecutor(risk, PaperForexBroker(specs), pf, specs)
    state = WebhookState(ex, SECRET)
    return TestClient(create_app(state)), pf, state


def alert(**kw):
    base = {"secret": SECRET, "symbol": "EURUSD", "action": "buy",
            "price": 1.1000, "atr": 0.0015}
    base.update(kw)
    return base


def test_secret_required(client):
    c, pf, _ = client
    r = c.post("/webhook", json=alert(secret="wrong" * 8))
    assert r.status_code == 403
    assert pf.positions == {}


def test_weak_server_secret_refused():
    with pytest.raises(ValueError, match="16 chars"):
        WebhookState(executor=None, secret="short")


def test_schema_rejects_garbage(client):
    c, *_ = client
    assert c.post("/webhook", json={"secret": SECRET}).status_code == 422
    assert c.post("/webhook",
                  json=alert(action="yolo")).status_code == 422
    assert c.post("/webhook", json=alert(price=-5)).status_code == 422


def test_unknown_symbol_rejected(client):
    c, pf, _ = client
    r = c.post("/webhook", json=alert(symbol="DOGEUSD"))
    assert r.status_code == 400
    assert pf.positions == {}


def test_buy_alert_opens_risk_sized_position(client):
    c, pf, _ = client
    r = c.post("/webhook", json=alert())
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "EURUSD" in pf.positions
    pos = pf.positions["EURUSD"]
    # risk engine, not TradingView, chose the size: ~1% risk at 2xATR stop
    risk_dollars = abs(pos.entry_price - pos.stop_price) * abs(pos.qty)
    assert risk_dollars <= 100_000 * 0.01 * 1.05


def test_cooldown_blocks_alert_storm(client):
    c, pf, state = client
    assert c.post("/webhook", json=alert()).json()["status"] == "ok"
    r2 = c.post("/webhook", json=alert())
    assert r2.json()["status"] in ("ignored", "rejected")
    # duplicate within cooldown never doubles the position
    assert len(pf.positions) == 1
    state.last_alert["EURUSD"] -= COOLDOWN_SECONDS + 1   # expire cooldown
    r3 = c.post("/webhook", json=alert())
    assert ("already open" in r3.json()["reason"]
            or r3.json()["reason"] == "filled")


def test_close_flow(client):
    c, pf, _ = client
    c.post("/webhook", json=alert())
    assert "EURUSD" in pf.positions
    r = c.post("/webhook", json=alert(action="close", price=1.1010))
    assert r.json()["status"] == "ok"
    assert pf.positions == {}
    r2 = c.post("/webhook", json=alert(action="close", price=1.1010))
    assert r2.json()["status"] == "ignored"


def test_missing_atr_uses_conservative_default(client):
    c, pf, _ = client
    r = c.post("/webhook", json={k: v for k, v in alert().items()
                                 if k != "atr"})
    assert r.status_code == 200
    pos = pf.positions["EURUSD"]
    stop_dist = abs(pos.entry_price - pos.stop_price)
    assert stop_dist >= 1.1000 * 0.003 * 2 * 0.99   # wide stop, small size


def test_halted_engine_rejects_alerts(client):
    c, pf, state = client
    state.executor.risk.core._halted.set()
    r = c.post("/webhook", json=alert())
    assert r.json()["status"] == "rejected"
    assert "halted" in r.json()["reason"] or "breaker" in r.json()["reason"]
    assert pf.positions == {}


def test_health_endpoint(client):
    c, *_ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["halted"] is False
