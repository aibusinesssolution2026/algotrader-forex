"""Configuration sweep for the multi-timeframe swarm.

Design decisions that matter:

1. SIGNAL CACHING. Agent predictions depend only on (timeframe, timestamp),
   not on the confluence config — so each agent's signal series is computed
   ONCE and every config in the grid just recombines cached votes. This
   makes a 48-config sweep cost barely more than one evaluation.

2. WALK-FORWARD, THREE WINDOWS. Models train on TRAIN, configs are RANKED
   on VALIDATION, and the winners are re-scored on a final untouched TEST
   window. Sweeping 48 configs is 48 chances to get lucky; the
   validation->test gap ("shrinkage") is reported so overfitting is visible
   instead of hidden. A config that ranks #1 on validation but flops on test
   was luck, not skill.

3. Ranking metric: net return after spread costs (not hit-rate — a config
   can hit 60% and still lose to spreads).
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import pandas as pd

from ..backtest.engine import Backtester, BacktestResult
from ..config import BacktestConfig
from .multitf import TimeframeAgent, tf_minutes

log = logging.getLogger(__name__)

DEFAULT_GRID = {
    "timeframe_sets": [
        ("1min", "3min", "5min", "15min", "30min", "60min", "120min", "180min"),
        ("1min", "3min", "5min", "15min"),                    # fast only
        ("30min", "60min", "120min", "180min"),               # slow only
        ("5min", "15min", "60min", "180min"),                 # classic ladder
    ],
    "n_anchors": [1, 2, 3],
    "min_agreement": [0.5, 0.6, 0.75, 0.9],
}


@dataclass(frozen=True)
class SwarmConfig:
    timeframes: tuple[str, ...]
    n_anchors: int
    min_agreement: float

    def label(self) -> str:
        tfs = "/".join(t.replace("min", "m") for t in self.timeframes)
        return f"[{tfs}] anchors={self.n_anchors} agree>={self.min_agreement:.0%}"


@dataclass
class ConfigScore:
    config: SwarmConfig
    val: BacktestResult
    test: BacktestResult | None = None

    @property
    def val_return(self) -> float:
        return self.val.total_return


def grid_configs(grid: dict | None = None) -> list[SwarmConfig]:
    g = grid or DEFAULT_GRID
    out = []
    for tfs, na, ma in itertools.product(g["timeframe_sets"], g["n_anchors"],
                                         g["min_agreement"]):
        if na <= len(tfs):
            out.append(SwarmConfig(tuple(sorted(tfs, key=tf_minutes)),
                                   na, ma))
    return out


def cache_agent_signals(base: pd.DataFrame, timeframes: list[str],
                        eval_index: pd.DatetimeIndex,
                        train_frac: float) -> pd.DataFrame:
    """One column of {-1,0,1} per timeframe, evaluated at eval_index.
    Each agent computes its per-bar signal once, then it is forward-mapped
    onto eval timestamps (causally: last fully closed bar)."""
    cols = {}
    for tf in sorted(set(timeframes), key=tf_minutes):
        agent = TimeframeAgent(tf)
        if not agent.prepare(base, train_frac=train_frac):
            continue
        per_bar = pd.Series(
            [agent.signal_at(bar_ts + pd.Timedelta(tf))
             for bar_ts in agent.feats.index],
            index=agent.feats.index + pd.Timedelta(tf))
        cols[tf] = per_bar.reindex(eval_index, method="ffill").fillna(0) \
            .astype(int)
        log.info("cached %s: %d bars", tf, len(per_bar))
    return pd.DataFrame(cols, index=eval_index)


def confluence_signal(votes: pd.DataFrame, cfg: SwarmConfig) -> pd.Series:
    """Vectorized confluence over cached votes (mirrors MultiTimeframeSwarm)."""
    tfs = [t for t in cfg.timeframes if t in votes.columns]
    if not tfs:
        return pd.Series(0, index=votes.index)
    anchors = tfs[-min(cfg.n_anchors, len(tfs)):]
    w = pd.Series({t: float(tf_minutes(t)) for t in tfs})
    v = votes[tfs]

    anchor_sum = v[anchors].sum(axis=1)
    anchor_active = (v[anchors] != 0).any(axis=1)
    direction = anchor_sum.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    veto = pd.Series(False, index=v.index)
    for a in anchors:
        veto |= (v[a] == -direction) & (direction != 0)

    agree_w = pd.Series(0.0, index=v.index)
    for t in tfs:
        agree_w += w[t] * (v[t] == direction) * (direction != 0)
    agreement = agree_w / w.sum()

    sig = direction.where(anchor_active & ~veto
                          & (agreement >= cfg.min_agreement), 0)
    return sig.astype(int)


def run_sweep(base: pd.DataFrame, spread_bps: float,
              grid: dict | None = None,
              splits: tuple[float, float] = (0.6, 0.8),
              top_k: int = 5) -> list[ConfigScore]:
    """Train on [0, s0), rank on [s0, s1), final-score top_k on [s1, end)."""
    configs = grid_configs(grid)
    all_tfs = sorted({t for c in configs for t in c.timeframes},
                     key=tf_minutes)
    s0, s1 = (base.index[int(len(base) * f)] for f in splits)

    from .multitf import resample_ohlcv
    from ..data.preprocess import engineer_features
    eval_bars = engineer_features(resample_ohlcv(base, "5min"))
    val_bars = eval_bars[(eval_bars.index >= s0) & (eval_bars.index < s1)]
    test_bars = eval_bars[eval_bars.index >= s1]

    votes = cache_agent_signals(base, all_tfs, eval_bars.index,
                                train_frac=splits[0])
    bt = Backtester(BacktestConfig(slippage_bps=spread_bps / 2, fee_bps=0.0,
                                   min_fee=0.0))

    def score(bars: pd.DataFrame, sig: pd.Series) -> BacktestResult:
        return bt.run(bars, lambda ts, row, s: int(sig.loc[ts]))

    results = []
    for cfg in configs:
        sig = confluence_signal(votes, cfg)
        results.append(ConfigScore(cfg, val=score(val_bars, sig)))
    results.sort(key=lambda r: r.val_return, reverse=True)

    for r in results[:top_k]:
        sig = confluence_signal(votes, r.config)
        r.test = score(test_bars, sig)
    return results


def report(results: list[ConfigScore], top_k: int = 5) -> str:
    lines = [f"{'rank':>4}  {'val ret':>8}  {'test ret':>9}  "
             f"{'val trades':>10}  config"]
    for i, r in enumerate(results[:top_k], 1):
        test = f"{r.test.total_return:+.2%}" if r.test else "   -"
        lines.append(f"{i:>4}  {r.val_return:+8.2%}  {test:>9}  "
                     f"{len(r.val.trades):>10}  {r.config.label()}")
    if results and results[0].test is not None:
        gap = results[0].val_return - results[0].test.total_return
        lines.append(f"\nOverfit gap (rank-1 val minus test): {gap:+.2%}"
                     "\n  ~0        -> config generalizes so far"
                     "\n  large +   -> validation rank was luck; distrust it")
    return "\n".join(lines)
