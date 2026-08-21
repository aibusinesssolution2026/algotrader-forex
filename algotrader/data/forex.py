"""Forex market layer.

- Pair metadata: pip size, quote conventions (JPY pairs use 0.01 pips).
- Data: Yahoo Finance forex tickers ('EURUSD=X') with the same cleaning
  pipeline as equities; deterministic synthetic FX generator for offline use
  (lower vol, mild mean reversion — closer to real FX behavior than GBM).
- Session calendar: forex trades 24/5, Sunday 21:00 UTC -> Friday 21:00 UTC.
  The trading loop must refuse orders in the weekend gap.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MAJOR_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
               "NZDUSD"]


@dataclass(frozen=True)
class PairSpec:
    symbol: str
    pip_size: float          # 0.0001 for most, 0.01 for JPY quotes
    typical_spread_pips: float


def pair_spec(symbol: str, spread_pips: float | None = None) -> PairSpec:
    symbol = symbol.upper().replace("=X", "").replace("/", "")
    is_jpy = symbol.endswith("JPY")
    default_spread = 1.5 if symbol in ("EURUSD", "USDJPY") else 2.5
    return PairSpec(symbol=symbol,
                    pip_size=0.01 if is_jpy else 0.0001,
                    typical_spread_pips=spread_pips or default_spread)


def to_pips(spec: PairSpec, price_delta: float) -> float:
    return price_delta / spec.pip_size


def yahoo_symbol(pair: str) -> str:
    return pair.upper().replace("/", "").replace("=X", "") + "=X"


def fetch_forex(pairs: list[str], period: str = "2y",
                interval: str = "1h") -> dict[str, pd.DataFrame]:
    """Fetch forex OHLC from Yahoo. Volume is synthetic/zero on Yahoo FX
    feeds, so we fill it with 1.0 to keep the pipeline uniform."""
    from .fetcher import fetch_yahoo
    raw = fetch_yahoo([yahoo_symbol(p) for p in pairs], period, interval)
    out = {}
    for p in pairs:
        key = yahoo_symbol(p)
        if key in raw:
            df = raw[key]
            df["volume"] = df["volume"].replace(0, 1.0).fillna(1.0)
            out[pair_spec(p).symbol] = df
    return out


def synthetic_fx(pair: str = "EURUSD", n: int = 2000, seed: int = 7,
                 freq: str = "h") -> pd.DataFrame:
    """Deterministic Ornstein-Uhlenbeck-flavored FX series: small drift,
    mean reversion toward a slowly wandering anchor, realistic hourly vol."""
    spec = pair_spec(pair)
    base = 150.0 if spec.pip_size == 0.01 else 1.10
    rng = np.random.default_rng(abs(hash(pair)) % (2**32) ^ seed)
    hourly_vol = base * 0.0009
    anchor = base + np.cumsum(rng.normal(0, hourly_vol / 6, n))
    px = np.empty(n)
    px[0] = base
    for i in range(1, n):
        px[i] = px[i - 1] + 0.03 * (anchor[i] - px[i - 1]) \
            + rng.normal(0, hourly_vol)
    spread = np.abs(rng.normal(0, hourly_vol / 2, n))
    open_ = np.concatenate([[base], px[:-1]])
    high = np.maximum(open_, px) + spread
    low = np.minimum(open_, px) - spread
    idx = pd.date_range("2024-01-01 21:00", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": px, "volume": np.ones(n)}, index=idx)


# ---------------- 24/5 session calendar ----------------

def market_open(ts: datetime | None = None) -> bool:
    """Forex is open Sunday 21:00 UTC through Friday 21:00 UTC."""
    ts = ts or datetime.now(timezone.utc)
    ts = ts.astimezone(timezone.utc)
    wd, hour = ts.weekday(), ts.hour  # Mon=0 .. Sun=6
    if wd == 5:                                   # Saturday: closed
        return False
    if wd == 6:                                   # Sunday: opens 21:00
        return hour >= 21
    if wd == 4:                                   # Friday: closes 21:00
        return hour < 21
    return True                                   # Mon-Thu: open


def rollover_window(ts: datetime | None = None) -> bool:
    """~22:00 UTC daily rollover: spreads blow out; avoid new entries."""
    ts = (ts or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return ts.hour == 22 and market_open(ts)
