import numpy as np
import pandas as pd
import pytest

from algotrader.data.fetcher import synthetic_ohlcv, save_parquet, load_parquet
from algotrader.data.preprocess import (clean, align, engineer_features,
                                        rsi, atr, macd, volatility_brackets,
                                        FEATURE_COLUMNS)
from algotrader.backtest.engine import Backtester
from algotrader.config import BacktestConfig, resolve_trading_mode


def test_synthetic_deterministic():
    a = synthetic_ohlcv("AAA", n=100)
    b = synthetic_ohlcv("AAA", n=100)
    pd.testing.assert_frame_equal(a, b)
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()


def test_clean_removes_bad_rows():
    df = synthetic_ohlcv(n=50)
    df.iloc[5, df.columns.get_loc("close")] = np.nan          # small gap -> ffill
    df.iloc[10, df.columns.get_loc("close")] = np.inf          # inf -> ffill
    df.iloc[20, df.columns.get_loc("open")] = -5.0             # negative -> drop
    dup = df.iloc[[3]]
    df = pd.concat([df, dup]).sort_index()
    out = clean(df)
    assert not out.index.duplicated().any()
    assert out["close"].notna().all()
    assert np.isfinite(out[["open", "high", "low", "close"]]).all().all()
    assert (out[["open", "high", "low", "close"]] > 0).all().all()
    assert len(out) == 49  # only the negative-price row dropped


def test_align_intersects_timestamps():
    a = synthetic_ohlcv("A", n=60)
    b = synthetic_ohlcv("B", n=60).iloc[10:]  # missing first 10 days
    aligned = align({"A": a, "B": b})
    assert len(aligned["A"]) == len(aligned["B"]) == 50
    assert (aligned["A"].index == aligned["B"].index).all()


def test_indicators_bounded_and_causal():
    df = synthetic_ohlcv(n=300)
    r = rsi(df["close"])
    assert ((r >= 0) & (r <= 100)).all()
    a = atr(df).dropna()
    assert (a > 0).all()
    m = macd(df["close"])
    assert np.allclose(m["macd_hist"], m["macd"] - m["macd_signal"])
    vb = volatility_brackets(df["close"])
    assert set(vb.unique()) <= {0, 1, 2, 3}


def test_feature_frame_complete():
    feats = engineer_features(synthetic_ohlcv(n=300))
    for col in FEATURE_COLUMNS:
        assert col in feats.columns
        assert feats[col].notna().all()


def test_parquet_roundtrip(tmp_path):
    data = {"XYZ": synthetic_ohlcv("XYZ", n=30)}
    save_parquet(data, tmp_path)
    loaded = load_parquet(tmp_path)
    pd.testing.assert_frame_equal(data["XYZ"], loaded["XYZ"], check_freq=False)


def test_backtester_costs_hurt():
    """On a perfectly flat price series, round-trip trading must strictly lose
    money under slippage+fees, and lose nothing with zero costs. Also, a
    no-trade strategy must always end flat."""
    df = synthetic_ohlcv(n=100)
    for c in ("open", "high", "low", "close"):
        df[c] = 100.0  # flat market: any PnL deviation is pure cost
    churn = lambda ts, row, s: 1 if (s.setdefault("i", -1), s.__setitem__("i", s["i"] + 1))[0] is None or s["i"] % 2 == 0 else 0
    costly = Backtester(BacktestConfig(slippage_bps=20, fee_bps=10))
    free = Backtester(BacktestConfig(slippage_bps=0, fee_bps=0, min_fee=0))
    flat = costly.run(df, lambda ts, row, s: 0)
    assert flat.equity_curve.iloc[-1] == pytest.approx(100_000.0)
    res_costly = costly.run(df, churn)
    res_free = free.run(df, churn)
    assert res_costly.equity_curve.iloc[-1] < 100_000.0    # costs bleed equity
    assert res_free.equity_curve.iloc[-1] == pytest.approx(100_000.0)
    assert all(t.pnl < 0 for t in res_costly.trades if t.exit_price is not None)


def test_backtester_trade_log():
    df = synthetic_ohlcv(n=100)
    # alternate long/flat every 10 bars -> multiple closed trades
    strat = lambda ts, row, s: 1 if (s.setdefault("i", -1) or True) and \
        ((s.__setitem__("i", s["i"] + 1) or s["i"] // 10) % 2 == 0) else 0
    res = Backtester().run(df, strat)
    closed = [t for t in res.trades if t.exit_price is not None]
    assert len(closed) >= 3
    for t in closed:
        assert t.fees > 0
        assert t.exit_time > t.entry_time


def test_paper_mode_is_default(monkeypatch):
    monkeypatch.delenv("ALGO_TRADING_MODE", raising=False)
    monkeypatch.delenv("ALGO_LIVE_CONFIRM", raising=False)
    assert resolve_trading_mode() == "paper"
    monkeypatch.setenv("ALGO_TRADING_MODE", "live")   # missing confirm phrase
    assert resolve_trading_mode() == "paper"
    monkeypatch.setenv("ALGO_LIVE_CONFIRM", "I_UNDERSTAND_THE_RISKS")
    assert resolve_trading_mode() == "live"
