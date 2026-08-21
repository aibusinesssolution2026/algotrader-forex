"""Multi-timeframe agent swarm.

One agent per timeframe (1m, 3m, 5m, 15m, 30m, 1h, 2h, 3h — configurable).
Each agent sees ONLY bars of its own timeframe, resampled causally from the
base feed, and emits {-1, 0, +1}.

Confluence aggregation (how the votes combine):
  1. TREND GATE: the slowest ("anchor") timeframes define the allowed
     direction. Lower-TF agents can only *time entries* in that direction,
     never fight it. This is the classic "trade with the higher timeframe"
     rule and is the main false-signal filter.
  2. WEIGHTED VOTE: agents vote with weights proportional to their timeframe
     (slower = more weight). A trade fires only when the agreeing weight
     fraction exceeds `min_agreement` (default 60%).
  3. VETO: if any anchor agent disagrees with the proposed direction,
     the signal is killed.

Resampling is strictly causal: at time T, an agent's latest bar is the last
FULLY CLOSED bar of its timeframe at or before T — no peeking into a bar
still forming.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ..data.preprocess import engineer_features
from .ensemble import StatisticalFallback, macd_convergence_signal

log = logging.getLogger(__name__)

DEFAULT_TIMEFRAMES = ["1min", "3min", "5min", "15min", "30min",
                      "60min", "120min", "180min"]

_OHLC_AGG = {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule, label="left", closed="left").agg(_OHLC_AGG)
    return out.dropna(subset=["close"])


def tf_minutes(rule: str) -> int:
    return int(pd.Timedelta(rule).total_seconds() // 60)


@dataclass
class TimeframeAgent:
    """Owns one timeframe: resamples, engineers features, predicts."""
    timeframe: str
    use_model: bool = True                 # False -> pure MACD rule
    model: StatisticalFallback | None = field(default=None, repr=False)
    feats: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def weight(self) -> float:
        return float(tf_minutes(self.timeframe))

    def prepare(self, base_df: pd.DataFrame, train_frac: float = 0.7) -> bool:
        bars = resample_ohlcv(base_df, self.timeframe)
        if len(bars) < 120:                # not enough bars to be meaningful
            log.info("Agent %s: only %d bars; disabled",
                     self.timeframe, len(bars))
            return False
        self.feats = engineer_features(bars)
        if self.use_model and len(self.feats) >= 150:
            self.model = StatisticalFallback()
            self.model.fit(self.feats.iloc[:int(len(self.feats) * train_frac)])
        return True

    def latest_closed_row(self, ts: pd.Timestamp) -> pd.Series | None:
        """Last feature row whose bar FULLY CLOSED at or before ts."""
        if self.feats is None:
            return None
        cutoff = ts - pd.Timedelta(self.timeframe)
        idx = self.feats.index[self.feats.index <= cutoff]
        if len(idx) == 0:
            return None
        return self.feats.loc[idx[-1]]

    def signal_at(self, ts: pd.Timestamp) -> int:
        row = self.latest_closed_row(ts)
        if row is None:
            return 0
        if self.model is not None:
            try:
                return self.model.predict(row)
            except Exception:
                pass
        return macd_convergence_signal(row)


@dataclass
class ConfluenceDecision:
    signal: int
    agreement: float                       # agreeing weight / total weight
    votes: dict[str, int]
    reason: str


class MultiTimeframeSwarm:
    def __init__(self, timeframes: list[str] | None = None,
                 n_anchors: int = 2, min_agreement: float = 0.6):
        tfs = sorted(timeframes or DEFAULT_TIMEFRAMES, key=tf_minutes)
        self.agents = [TimeframeAgent(tf) for tf in tfs]
        self.n_anchors = n_anchors
        self.min_agreement = min_agreement

    def prepare(self, base_df: pd.DataFrame, train_frac: float = 0.7) -> None:
        self.agents = [a for a in self.agents
                       if a.prepare(base_df, train_frac)]
        if not self.agents:
            raise ValueError("No timeframe has enough data")
        log.info("Swarm active agents: %s",
                 [a.timeframe for a in self.agents])

    @property
    def anchors(self) -> list[TimeframeAgent]:
        return self.agents[-min(self.n_anchors, len(self.agents)):]

    def decide(self, ts: pd.Timestamp) -> ConfluenceDecision:
        votes = {a.timeframe: a.signal_at(ts) for a in self.agents}

        # 1. trend gate from anchors
        anchor_votes = [votes[a.timeframe] for a in self.anchors]
        directional = [v for v in anchor_votes if v != 0]
        if not directional:
            return ConfluenceDecision(0, 0.0, votes, "anchors flat")
        direction = 1 if sum(directional) > 0 else -1
        # 3. anchor veto: any anchor actively opposing kills the trade
        if any(v == -direction for v in anchor_votes):
            return ConfluenceDecision(0, 0.0, votes, "anchor veto")

        # 2. weighted agreement across ALL agents
        total_w = sum(a.weight for a in self.agents)
        agree_w = sum(a.weight for a in self.agents
                      if votes[a.timeframe] == direction)
        agreement = agree_w / total_w
        if agreement < self.min_agreement:
            return ConfluenceDecision(
                0, agreement, votes,
                f"agreement {agreement:.0%} < {self.min_agreement:.0%}")
        return ConfluenceDecision(direction, agreement, votes, "confluence")
