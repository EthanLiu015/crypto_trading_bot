"""Feature engineering from OHLCV DataFrames (pure pandas/numpy)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _atr(high, low, close, length)
    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(length).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(length).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(length).mean()


def _hurst_rs(series: np.ndarray) -> float:
    n = len(series)
    if n < 20:
        return np.nan
    mean = series.mean()
    deviate = np.cumsum(series - mean)
    r = deviate.max() - deviate.min()
    s = series.std(ddof=1)
    if s == 0 or r == 0:
        return np.nan
    return float(np.log(r / s) / np.log(n))


def _rolling_hurst(log_returns: pd.Series, window: int = 100) -> pd.Series:
    return log_returns.rolling(window, min_periods=window).apply(
        lambda x: _hurst_rs(np.asarray(x)), raw=False
    )


def _rolling_autocorr(log_returns: pd.Series, window: int = 20, lag: int = 1) -> pd.Series:
    def ac1(x: pd.Series) -> float:
        arr = np.asarray(x)
        if len(arr) < lag + 2:
            return np.nan
        a, b = arr[:-lag], arr[lag:]
        if a.std() == 0 or b.std() == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    return log_returns.rolling(window, min_periods=window).apply(ac1, raw=False)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features from OHLCV.

    Expects columns: open, high, low, close, volume (timestamp optional).
    """
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    out["adx"] = _adx(high, low, close, 14)
    atr = _atr(high, low, close, 14)
    out["atr_pct"] = (atr / close).replace([np.inf, -np.inf], np.nan)

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    out["bb_lower"] = lower
    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_width"] = (upper - lower) / mid.replace(0, np.nan)

    out["rsi"] = _rsi(close, 14)

    log_ret = np.log(close / close.shift(1))
    short_w = min(120, max(5, len(out) // 10))
    long_w = min(720, max(30, len(out) // 3))
    rv_short = log_ret.rolling(short_w, min_periods=5).std()
    rv_long = log_ret.rolling(long_w, min_periods=10).std()
    out["vol_ratio"] = rv_short / rv_long.replace(0, np.nan)

    out["autocorr_1"] = _rolling_autocorr(log_ret, window=20, lag=1)
    out["hurst"] = _rolling_hurst(log_ret, window=min(100, max(50, len(out) // 5)))

    out["ema_9"] = _ema(close, 9)
    out["ema_21"] = _ema(close, 21)

    out["high_20"] = high.rolling(20, min_periods=20).max().shift(1)
    out["low_20"] = low.rolling(20, min_periods=20).min().shift(1)

    return out
