import pandas as pd
import pytest

from algotrader.data.forex import synthetic_fx
from algotrader.models.multitf import MultiTimeframeSwarm
from algotrader.models.sweep import (grid_configs, cache_agent_signals,
                                     confluence_signal, run_sweep, report,
                                     SwarmConfig, DEFAULT_GRID)


@pytest.fixture(scope="module")
def base():
    return synthetic_fx("EURUSD", n=8000, freq="min")


def test_grid_size_and_validity():
    cfgs = grid_configs()
    assert len(cfgs) == 48
    for c in cfgs:
        assert c.n_anchors <= len(c.timeframes)
        assert 0 < c.min_agreement <= 1


def test_cached_votes_match_live_swarm(base):
    """The vectorized cached path must agree with the live object path —
    otherwise the sweep would rank a different system than we deploy."""
    tfs = ["5min", "15min", "60min"]
    eval_idx = base.index[6000:7000:25]
    votes = cache_agent_signals(base, tfs, pd.DatetimeIndex(eval_idx),
                                train_frac=0.6)
    swarm = MultiTimeframeSwarm(timeframes=tfs, n_anchors=2,
                                min_agreement=0.6)
    swarm.prepare(base, train_frac=0.6)
    cfg = SwarmConfig(tuple(tfs), 2, 0.6)
    cached_sig = confluence_signal(votes, cfg)
    mismatches = sum(
        1 for ts in eval_idx
        if swarm.decide(ts).signal != int(cached_sig.loc[ts]))
    assert mismatches / len(eval_idx) < 0.10   # allow boundary-bar edge cases


def test_confluence_vector_semantics(base):
    idx = pd.DatetimeIndex(base.index[:4])
    votes = pd.DataFrame({"5min": [1, 1, 1, 0], "60min": [1, 0, -1, 0],
                          "180min": [1, 1, -1, 0]}, index=idx)
    cfg = SwarmConfig(("5min", "60min", "180min"), 2, 0.6)
    sig = confluence_signal(votes, cfg)
    assert sig.iloc[0] == 1     # unanimous
    assert sig.iloc[1] == 1     # anchors: one long one flat -> long, no veto
    assert sig.iloc[2] == -1    # anchors unite short; non-anchor 5min has no veto
    assert sig.iloc[3] == 0     # everyone flat


def test_run_sweep_walk_forward(base):
    grid = {"timeframe_sets": [("5min", "15min", "60min")],
            "n_anchors": [1, 2], "min_agreement": [0.5, 0.9]}
    results = run_sweep(base, spread_bps=1.3, grid=grid, top_k=2)
    assert len(results) == 4
    # sorted by validation return
    vals = [r.val_return for r in results]
    assert vals == sorted(vals, reverse=True)
    # only top_k got test scores; test window untouched for the rest
    assert results[0].test is not None and results[1].test is not None
    assert all(r.test is None for r in results[2:])
    rep = report(results, top_k=2)
    assert "Overfit gap" in rep and "config" in rep


def test_report_handles_empty():
    assert report([]) != ""
