import numpy as np
import pandas as pd
import pytest

from algotrader.data.forex import synthetic_fx
from algotrader.models.multitf import (MultiTimeframeSwarm, TimeframeAgent,
                                       resample_ohlcv, tf_minutes,
                                       ConfluenceDecision)


@pytest.fixture(scope="module")
def base_1min():
    # ~10 days of 1-minute FX data
    return synthetic_fx("EURUSD", n=14_000, freq="min")


def test_tf_minutes():
    assert tf_minutes("1min") == 1
    assert tf_minutes("180min") == 180


def test_resample_ohlc_semantics(base_1min):
    m5 = resample_ohlcv(base_1min, "5min")
    window = base_1min.iloc[0:5]
    assert m5.iloc[0]["open"] == window["open"].iloc[0]
    assert m5.iloc[0]["close"] == window["close"].iloc[-1]
    assert m5.iloc[0]["high"] == window["high"].max()
    assert m5.iloc[0]["low"] == window["low"].min()
    assert m5.iloc[0]["volume"] == pytest.approx(window["volume"].sum())


def test_agent_never_sees_forming_bar(base_1min):
    """Causality: at time T the newest usable bar must have CLOSED by T."""
    agent = TimeframeAgent("30min", use_model=False)
    assert agent.prepare(base_1min)
    ts = base_1min.index[5000]
    row = agent.latest_closed_row(ts)
    bar_open = row.name
    bar_close = bar_open + pd.Timedelta("30min")
    assert bar_close <= ts


def test_swarm_prepare_drops_thin_timeframes():
    short = synthetic_fx("EURUSD", n=600, freq="min")  # 10 hours of data
    swarm = MultiTimeframeSwarm()
    swarm.prepare(short)
    tfs = [a.timeframe for a in swarm.agents]
    assert "1min" in tfs
    assert "180min" not in tfs      # only ~3 bars -> disabled


class FixedAgent(TimeframeAgent):
    """Test double: always votes a fixed signal."""
    def __init__(self, timeframe, sig):
        super().__init__(timeframe=timeframe, use_model=False)
        self._sig = sig
        self.feats = pd.DataFrame()   # marks as prepared

    def signal_at(self, ts):
        return self._sig


def make_swarm(votes: dict[str, int], n_anchors=2, min_agreement=0.6):
    swarm = MultiTimeframeSwarm(timeframes=list(votes), n_anchors=n_anchors,
                                min_agreement=min_agreement)
    swarm.agents = [FixedAgent(tf, sig) for tf, sig in
                    sorted(votes.items(), key=lambda kv: tf_minutes(kv[0]))]
    return swarm


TS = pd.Timestamp("2026-07-13 12:00", tz="UTC")


def test_full_confluence_fires():
    d = make_swarm({"5min": 1, "15min": 1, "60min": 1, "180min": 1}).decide(TS)
    assert d.signal == 1 and d.agreement == pytest.approx(1.0)


def test_anchor_veto_kills_signal():
    # fast TFs scream long, but the 180min anchor is short -> veto
    d = make_swarm({"5min": 1, "15min": 1, "60min": 1, "180min": -1}).decide(TS)
    assert d.signal == 0
    assert d.reason == "anchor veto"


def test_flat_anchors_no_trade():
    d = make_swarm({"5min": 1, "15min": 1, "60min": 0, "180min": 0}).decide(TS)
    assert d.signal == 0
    assert d.reason == "anchors flat"


def test_agreement_threshold():
    # anchors long, but every fast TF disagrees: weight fraction below 60%
    votes = {"1min": -1, "3min": -1, "5min": -1, "15min": -1, "30min": -1,
             "60min": 1, "120min": 1, "180min": 1}
    d = make_swarm(votes, n_anchors=3).decide(TS)
    # agreeing weight = 60+120+180=360 of 414 total = 87% -> fires; use
    # tighter threshold to verify the gate itself
    d2 = make_swarm(votes, n_anchors=3, min_agreement=0.95).decide(TS)
    assert d.signal == 1
    assert d2.signal == 0 and "agreement" in d2.reason


def test_weighting_favors_slow_timeframes():
    # equal head-count split, but slow TFs carry the weight -> their side wins
    votes = {"1min": -1, "3min": -1, "5min": -1,
             "60min": 1, "120min": 1, "180min": 1}
    d = make_swarm(votes, n_anchors=2, min_agreement=0.5).decide(TS)
    assert d.signal == 1


def test_swarm_end_to_end_reduces_trade_count(base_1min):
    """The core claim: confluence trades LESS than a single fast agent.
    Fewer trades = less spread bleed. Verified, not assumed."""
    swarm = MultiTimeframeSwarm(
        timeframes=["5min", "15min", "60min", "180min"])
    swarm.prepare(base_1min)
    solo = TimeframeAgent("5min", use_model=False)
    solo.prepare(base_1min)

    sample_ts = base_1min.index[10_000:13_000:15]
    swarm_sigs = [swarm.decide(ts).signal for ts in sample_ts]
    solo_sigs = [solo.signal_at(ts) for ts in sample_ts]
    swarm_active = sum(s != 0 for s in swarm_sigs)
    solo_active = sum(s != 0 for s in solo_sigs)
    assert swarm_active < solo_active
    assert all(s in (-1, 0, 1) for s in swarm_sigs)
