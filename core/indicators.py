"""Technical analysis indicators — pure pandas/numpy."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(series, n)
    std = series.rolling(n, min_periods=max(2, n // 2)).std()
    upper = mid + k * std
    lower = mid - k * std
    width = (upper - lower) / mid
    return upper, mid, lower, width


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def annualized_vol(series: pd.Series, window: int = 30) -> float:
    ret = series.pct_change().dropna().tail(window)
    if len(ret) < 5:
        return float("nan")
    return float(ret.std() * np.sqrt(252))


def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    peak = series.cummax()
    dd = series / peak - 1.0
    return float(dd.min())


def pct_return(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) <= days:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-1 - days] - 1.0)
