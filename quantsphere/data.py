"""Stage 1 — market data ingestion and empirical statistics.

Pure functions (no Streamlit): caching happens at the app boundary so this
module stays testable and reusable from scripts or notebooks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def yf_download(tickers, retries: int = 3, **kwargs) -> pd.DataFrame:
    """yfinance download hardened against transient failures / rate limits."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(tickers, progress=False, **kwargs)
            if df is not None and not df.empty:
                return df
        except Exception as exc:  # network hiccups, JSON decode, rate limits
            last_exc = exc
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))
    detail = f" ({last_exc})" if last_exc else ""
    raise ValueError(f"No data returned for '{tickers}' after {retries} attempts{detail}")


def fetch_history(ticker: str, years: int = 5) -> pd.DataFrame:
    """Daily OHLCV, dividend/split adjusted, cleansed."""
    df = yf_download(ticker, period=f"{years}y", interval="1d", auto_adjust=True)
    df = _flatten(df)
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[cols]
    df = df[df["Close"] > 0]
    df = df.ffill().dropna()
    if len(df) < 60:
        raise ValueError(f"Insufficient history for '{ticker}' ({len(df)} rows)")
    return df


# Yahoo Finance lookback caps per intraday interval.
INTRADAY_PERIODS = {"1h": "730d", "30m": "60d", "15m": "60d", "5m": "60d", "1m": "7d"}


def fetch_bars(ticker: str, interval: str = "1d", years: int = 5) -> pd.DataFrame:
    """OHLCV bars at any supported timeframe, cleansed.

    Daily uses the full `years` window; intraday uses the maximum lookback
    Yahoo allows for that interval.
    """
    if interval == "1d":
        return fetch_history(ticker, years)
    if interval not in INTRADAY_PERIODS:
        raise ValueError(f"unsupported interval '{interval}'")
    df = fetch_intraday(ticker, INTRADAY_PERIODS[interval], interval)
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} bars of {interval} data for '{ticker}'")
    return df


def fetch_intraday(ticker: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
    """Latest intraday bars (yfinance caps 1m data at the last 7 days)."""
    df = yf_download(ticker, period=period, interval=interval, auto_adjust=True)
    df = _flatten(df)
    df = df[df["Close"] > 0].ffill().dropna()
    return df


@dataclass(frozen=True)
class MarketStats:
    last_price: float
    prev_close: float
    mu_annual: float          # empirical drift, annualized
    sigma_annual: float       # historical volatility, annualized
    skew: float
    kurtosis: float           # excess kurtosis
    var_95_daily: float       # historical 1-day VaR (loss, positive number)
    n_obs: int


def log_returns(close: pd.Series) -> pd.Series:
    lr = np.log(close / close.shift(1))
    return lr.dropna()


def rolling_volatility(close: pd.Series, window: int = 21) -> pd.Series:
    lr = log_returns(close)
    return lr.rolling(window).std() * np.sqrt(TRADING_DAYS)


def compute_stats(close: pd.Series) -> MarketStats:
    lr = log_returns(close)
    if len(lr) < 30:
        raise ValueError("Need at least 30 return observations")
    sigma_d = float(lr.std(ddof=1))
    return MarketStats(
        last_price=float(close.iloc[-1]),
        prev_close=float(close.iloc[-2]),
        mu_annual=float(lr.mean()) * TRADING_DAYS,
        sigma_annual=sigma_d * np.sqrt(TRADING_DAYS),
        skew=float(lr.skew()),
        kurtosis=float(lr.kurt()),
        var_95_daily=float(-np.quantile(lr, 0.05)),
        n_obs=int(len(lr)),
    )


def fetch_spot(ticker: str) -> float:
    """Best-effort live spot: fast_info, falling back to last daily close."""
    tk = yf.Ticker(ticker)
    try:
        px = float(tk.fast_info["last_price"])
        if np.isfinite(px) and px > 0:
            return px
    except (KeyError, TypeError, ValueError):
        pass
    hist = tk.history(period="5d", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"Cannot determine spot price for '{ticker}'")
    return float(hist["Close"].iloc[-1])


def fetch_option_chain(ticker: str, max_expiries: int = 8,
                       moneyness: tuple[float, float] = (0.6, 1.6)) -> tuple[pd.DataFrame, float]:
    """Options quotes across expiries: mid prices, filtered for liquidity.

    Returns (chain, spot). Chain columns: strike, T, mid, is_call, bid, ask,
    volume, open_interest, expiry.
    """
    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        raise ValueError(f"'{ticker}' has no listed options on Yahoo Finance")
    spot = fetch_spot(ticker)

    now = pd.Timestamp.utcnow().tz_localize(None)
    rows = []
    for exp in expiries[:max_expiries]:
        T = (pd.Timestamp(exp) + pd.Timedelta(hours=16) - now).total_seconds() / (365.0 * 86400)
        if T < 3.0 / 365.0:  # skip expiring-now contracts: IV explodes
            continue
        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue
        for frame, is_call in ((chain.calls, True), (chain.puts, False)):
            if frame is None or frame.empty:
                continue
            f = frame.copy()
            f["is_call"] = is_call
            f["T"] = T
            f["expiry"] = exp
            rows.append(f)

    if not rows:
        raise ValueError(f"No usable option expiries for '{ticker}'")
    df = pd.concat(rows, ignore_index=True)

    df["bid"] = pd.to_numeric(df.get("bid"), errors="coerce").fillna(0.0)
    df["ask"] = pd.to_numeric(df.get("ask"), errors="coerce").fillna(0.0)
    df["lastPrice"] = pd.to_numeric(df.get("lastPrice"), errors="coerce").fillna(0.0)
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
    df["openInterest"] = pd.to_numeric(df.get("openInterest"), errors="coerce").fillna(0.0)

    mid = np.where((df["bid"] > 0) & (df["ask"] > df["bid"]),
                   0.5 * (df["bid"] + df["ask"]), df["lastPrice"])
    df["mid"] = mid

    keep = (
        (df["mid"] > 0.01)
        & (df["strike"] >= moneyness[0] * spot)
        & (df["strike"] <= moneyness[1] * spot)
        & ((df["volume"] > 0) | (df["openInterest"] > 0))
    )
    out = df.loc[keep, ["strike", "T", "mid", "is_call", "bid", "ask",
                        "volume", "openInterest", "expiry"]].rename(
        columns={"openInterest": "open_interest"}).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"Option chain for '{ticker}' is entirely illiquid")
    return out, spot
