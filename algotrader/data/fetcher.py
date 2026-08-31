"""Historical data acquisition.

Primary source: Yahoo Finance via yfinance (daily bars; intraday down to 1m,
which is the closest 'tick-level' granularity a free API reliably provides).
Fallback: deterministic synthetic GBM generator so the whole platform can be
developed and tested offline / in CI without network access.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]


def fetch_yahoo(symbols: list[str], period: str = "2y",
                interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Fetch OHLCV from Yahoo Finance. interval: '1d' daily, '1m'/'5m' intraday."""
    import yfinance as yf  # imported lazily so offline envs never need it
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = yf.Ticker(sym).history(period=period, interval=interval,
                                    auto_adjust=True)
        if df.empty:
            log.warning("No data returned for %s", sym)
            continue
        df = df.rename(columns=str.lower)[OHLCV]
        df.index = pd.to_datetime(df.index, utc=True)
        out[sym] = df
    return out


def synthetic_ohlcv(symbol: str = "SYN", n: int = 500, seed: int = 42,
                    start_price: float = 100.0, freq: str = "D",
                    drift: float = 0.0002, vol: float = 0.015) -> pd.DataFrame:
    """Deterministic geometric-Brownian-motion OHLCV series for tests."""
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32) ^ seed)
    rets = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0, vol / 2, n)) * close
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(50_000, 500_000, n).astype(float)
    idx = pd.date_range("2023-01-02", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)


def save_parquet(data: dict[str, pd.DataFrame], directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for sym, df in data.items():
        df.to_parquet(directory / f"{sym}.parquet")


def load_parquet(directory: str | Path) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    return {p.stem: pd.read_parquet(p) for p in directory.glob("*.parquet")}
