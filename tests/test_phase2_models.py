import time

import pandas as pd
import pytest

from algotrader.data.fetcher import synthetic_ohlcv
from algotrader.data.preprocess import engineer_features
from algotrader.models.ensemble import (EnsemblePredictor, StatisticalFallback,
                                        macd_convergence_signal, build_labels)
from algotrader.models.evaluate import evaluate_predictor


@pytest.fixture(scope="module")
def feats():
    return engineer_features(synthetic_ohlcv(n=400))


@pytest.fixture(scope="module")
def fitted_fallback(feats):
    fb = StatisticalFallback()
    fb.fit(feats.iloc[:300])
    return fb


def test_labels_shape(feats):
    y = build_labels(feats)
    assert set(y.unique()) <= {0, 1}
    assert len(y) == len(feats)


def test_fallback_predicts_valid_signals(feats, fitted_fallback):
    sigs = {fitted_fallback.predict(feats.iloc[i]) for i in range(300, 350)}
    assert sigs <= {-1, 0, 1}


def test_primary_used_when_healthy(feats, fitted_fallback):
    with EnsemblePredictor(primary=lambda row: 1,
                           fallback=fitted_fallback) as ens:
        assert ens.predict(feats.iloc[-1]) == 1
        assert ens.tier_counts["primary"] == 1
        assert ens.tier_counts["statistical"] == 0


def test_fallback_on_primary_exception(feats, fitted_fallback):
    def broken(row):
        raise ConnectionError("429 rate limited")
    with EnsemblePredictor(primary=broken, fallback=fitted_fallback) as ens:
        sig = ens.predict(feats.iloc[-1])
        assert sig in (-1, 0, 1)
        assert ens.tier_counts["statistical"] == 1


def test_fallback_on_primary_timeout(feats, fitted_fallback):
    def slow(row):
        time.sleep(5)
        return 1
    with EnsemblePredictor(primary=slow, fallback=fitted_fallback,
                           timeout_s=0.2) as ens:
        t0 = time.monotonic()
        sig = ens.predict(feats.iloc[-1])
        assert time.monotonic() - t0 < 1.5  # did not wait 5s
        assert sig in (-1, 0, 1)
        assert ens.tier_counts["primary"] == 0


def test_macd_last_resort(feats):
    with EnsemblePredictor(primary=None, fallback=None) as ens:
        sig = ens.predict(feats.iloc[-1])
        assert sig == macd_convergence_signal(feats.iloc[-1])
        assert ens.tier_counts["macd"] == 1


def test_invalid_primary_signal_rejected(feats, fitted_fallback):
    with EnsemblePredictor(primary=lambda row: 7,
                           fallback=fitted_fallback) as ens:
        assert ens.predict(feats.iloc[-1]) in (-1, 0, 1)
        assert ens.tier_counts["primary"] == 0


def test_evaluation_metrics_perfect_oracle(feats):
    """An oracle that peeks at the future must score 100% hit-rate; this
    validates the metric plumbing."""
    fwd = feats["close"].shift(-1) / feats["close"] - 1
    oracle = lambda row: 1 if fwd.loc[row.name] > 0 else -1
    rep = evaluate_predictor(feats, oracle)
    assert rep.hit_rate == pytest.approx(1.0)
    assert rep.precision_long == pytest.approx(1.0)
    assert rep.precision_short == pytest.approx(1.0)
    assert rep.n_directional == rep.n_predictions


def test_evaluation_metrics_inverse_oracle(feats):
    fwd = feats["close"].shift(-1) / feats["close"] - 1
    anti = lambda row: -1 if fwd.loc[row.name] > 0 else 1
    rep = evaluate_predictor(feats, anti)
    assert rep.hit_rate == pytest.approx(0.0)


def test_evaluation_on_fitted_fallback(feats, fitted_fallback):
    rep = evaluate_predictor(feats.iloc[300:], fitted_fallback.predict)
    assert 0.0 <= rep.hit_rate <= 1.0
    assert rep.n_predictions == len(feats.iloc[300:]) - 1
