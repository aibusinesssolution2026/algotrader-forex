"""Quantitative evaluation of a predictor on historical feature frames:
accuracy, precision (long calls), and directional hit-rate on non-flat calls.
Walk-forward: the model only ever sees data strictly before the bar it
predicts, preventing look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class EvalReport:
    n_predictions: int
    n_directional: int          # predictions that were not flat (0)
    accuracy: float             # correct direction / all non-flat predictions
    precision_long: float       # of long calls, fraction that rose
    precision_short: float      # of short calls, fraction that fell
    hit_rate: float             # alias of accuracy (directional hit-rate)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def evaluate_predictor(feats: pd.DataFrame,
                       predict: Callable[[pd.Series], int],
                       horizon: int = 1) -> EvalReport:
    fwd_ret = feats["close"].shift(-horizon) / feats["close"] - 1
    rows = feats.iloc[:-horizon]

    total = long_hits = long_calls = short_hits = short_calls = 0
    for ts, row in rows.iterrows():
        sig = predict(row)
        total += 1
        if sig == 0:
            continue
        realized = fwd_ret.loc[ts]
        if sig == 1:
            long_calls += 1
            long_hits += int(realized > 0)
        else:
            short_calls += 1
            short_hits += int(realized < 0)

    directional = long_calls + short_calls
    hits = long_hits + short_hits
    return EvalReport(
        n_predictions=total,
        n_directional=directional,
        accuracy=hits / directional if directional else 0.0,
        precision_long=long_hits / long_calls if long_calls else 0.0,
        precision_short=short_hits / short_calls if short_calls else 0.0,
        hit_rate=hits / directional if directional else 0.0,
    )
