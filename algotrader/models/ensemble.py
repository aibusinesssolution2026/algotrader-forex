"""Ensemble predictive module with graceful degradation.

Tier 1 (primary):  external AI API predictor (pluggable callable). Any latency
                   timeout, rate-limit, or exception triggers fallback.
Tier 2 (fallback): gradient-boosted trees trained locally on engineered
                   features (XGBoost if installed, else sklearn's
                   GradientBoostingClassifier — same API, zero extra deps).
Tier 3 (last resort): pure MACD convergence signal — no model required.

All predictors emit a direction in {-1, 0, +1}.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from ..data.preprocess import FEATURE_COLUMNS

log = logging.getLogger(__name__)


def _make_boosted_model():
    try:  # optional heavy dependency
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=4,
                             learning_rate=0.05, eval_metric="logloss")
    except ImportError:
        return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                          learning_rate=0.05)


def build_labels(feats: pd.DataFrame, horizon: int = 1,
                 threshold: float = 0.0) -> pd.Series:
    """Label = 1 if forward return over `horizon` bars exceeds threshold,
    else 0. Rows without a full forward window are dropped by the caller."""
    fwd = feats["close"].shift(-horizon) / feats["close"] - 1
    return (fwd > threshold).astype(int)


class StatisticalFallback:
    """Tier-2 boosted-tree model on engineered features."""

    def __init__(self):
        self.model = _make_boosted_model()
        self.fitted = False

    def fit(self, feats: pd.DataFrame, horizon: int = 1) -> None:
        y = build_labels(feats, horizon)
        X = feats[FEATURE_COLUMNS].iloc[:-horizon]
        y = y.iloc[:-horizon]
        self.model.fit(X.values, y.values)
        self.fitted = True

    def predict(self, row: pd.Series) -> int:
        if not self.fitted:
            raise RuntimeError("Fallback model not fitted")
        x = row[FEATURE_COLUMNS].values.reshape(1, -1).astype(float)
        proba = float(self.model.predict_proba(x)[0, 1])
        if proba > 0.55:
            return 1
        if proba < 0.45:
            return -1
        return 0


def macd_convergence_signal(row: pd.Series) -> int:
    """Tier-3: long when MACD histogram positive and rising RSI regime,
    short on the mirror condition, flat otherwise."""
    if row["macd_hist"] > 0 and row["rsi_14"] > 50:
        return 1
    if row["macd_hist"] < 0 and row["rsi_14"] < 50:
        return -1
    return 0


class EnsemblePredictor:
    """Routes each prediction through the tiers with a hard latency budget."""

    def __init__(self,
                 primary: Callable[[pd.Series], int] | None = None,
                 fallback: StatisticalFallback | None = None,
                 timeout_s: float = 2.0):
        self.primary = primary
        self.fallback = fallback
        self.timeout_s = timeout_s
        self.tier_counts = {"primary": 0, "statistical": 0, "macd": 0}
        self._pool = cf.ThreadPoolExecutor(max_workers=1,
                                           thread_name_prefix="ai-primary")

    def predict(self, row: pd.Series) -> int:
        # Tier 1 — primary AI API under a strict timeout
        if self.primary is not None:
            fut = self._pool.submit(self.primary, row)
            try:
                sig = int(fut.result(timeout=self.timeout_s))
                if sig in (-1, 0, 1):
                    self.tier_counts["primary"] += 1
                    return sig
                log.warning("Primary returned invalid signal %s", sig)
            except cf.TimeoutError:
                fut.cancel()
                log.warning("Primary AI predictor timed out (> %.1fs)",
                            self.timeout_s)
            except Exception as exc:  # rate limits, HTTP errors, anything
                log.warning("Primary AI predictor failed: %s", exc)

        # Tier 2 — local boosted model
        if self.fallback is not None and self.fallback.fitted:
            try:
                self.tier_counts["statistical"] += 1
                return self.fallback.predict(row)
            except Exception as exc:
                log.warning("Statistical fallback failed: %s", exc)

        # Tier 3 — indicator rule, cannot fail
        self.tier_counts["macd"] += 1
        return macd_convergence_signal(row)

    def close(self) -> None:
        """Release the worker thread (call on shutdown to avoid leaks)."""
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
