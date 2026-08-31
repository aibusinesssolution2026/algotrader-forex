"""Cleaning, timestamp alignment, and technical feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fetcher import OHLCV


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate index, forward-fill small gaps, drop unfixable rows."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df[OHLCV] = df[OHLCV].replace([np.inf, -np.inf], np.nan)
    # forward-fill at most 2 consecutive missing frames, then drop the rest
    df[OHLCV] = df[OHLCV].ffill(limit=2)
    df = df.dropna(subset=["close"])
    # basic sanity: strictly positive prices, non-negative volume
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[df["volume"] >= 0]
    return df


def align(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every symbol to the intersection of timestamps (aligned panel)."""
    if not data:
        return data
    common = None
    for df in data.values():
        common = df.index if common is None else common.intersection(df.index)
    return {sym: df.loc[common].copy() for sym, df in data.items()}


# ---------------- technical indicators ----------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig,
                         "macd_hist": line - sig})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def volatility_brackets(close: pd.Series, window: int = 20,
                        n_brackets: int = 4) -> pd.Series:
    """Custom feature: bucket rolling realized volatility into regime brackets
    0 (calm) .. n_brackets-1 (turbulent), using expanding quantiles so the
    feature is causal (no look-ahead)."""
    rv = close.pct_change().rolling(window).std()
    ranks = rv.expanding(min_periods=window * 2).rank(pct=True)
    brackets = np.floor(ranks * n_brackets).clip(0, n_brackets - 1)
    return brackets.fillna(0).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the full feature set. Rows with warm-up NaNs are dropped."""
    out = df.copy()
    out["rsi_14"] = rsi(out["close"])
    out = out.join(macd(out["close"]))
    out["atr_14"] = atr(out)
    out["vol_bracket"] = volatility_brackets(out["close"])
    out["ret_1"] = out["close"].pct_change()
    out["ret_5"] = out["close"].pct_change(5)
    out["sma_ratio"] = out["close"] / out["close"].rolling(20).mean()
    return out.dropna()


FEATURE_COLUMNS = ["rsi_14", "macd", "macd_signal", "macd_hist", "atr_14",
                   "vol_bracket", "ret_1", "ret_5", "sma_ratio"]
