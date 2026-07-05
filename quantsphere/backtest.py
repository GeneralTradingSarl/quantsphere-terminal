"""Stage 6 — vectorized backtesting engine.

Execution discipline: a position computed with information up to bar t is
applied to the return of bar t+1 (one-bar execution lag), so no strategy can
look ahead. Costs are charged on turnover in basis points. All annualization
is timeframe-aware via `periods_per_year`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantsphere.engine import core

TRADING_DAYS = 252

# Bars per year per timeframe (US cash session ≈ 6.5 hours).
PERIODS_PER_YEAR = {
    "1d": 252.0,
    "1h": 252.0 * 6.5,
    "30m": 252.0 * 13.0,
    "15m": 252.0 * 26.0,
    "5m": 252.0 * 78.0,
}


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def perf_metrics(returns: pd.Series, rf: float = 0.0,
                 periods_per_year: float = TRADING_DAYS) -> dict:
    """Annualized performance statistics from per-bar simple returns."""
    returns = returns.dropna()
    if len(returns) < 20:
        raise ValueError("Need at least 20 return observations")
    ppy = float(periods_per_year)
    equity = (1.0 + returns).cumprod()
    years = len(returns) / ppy
    total = float(equity.iloc[-1])
    cagr = total ** (1.0 / years) - 1.0 if total > 0 else -1.0
    vol = float(returns.std(ddof=1)) * np.sqrt(ppy)
    mean_ann = float(returns.mean()) * ppy
    sharpe = (mean_ann - rf) / vol if vol > 0 else float("nan")
    downside = returns[returns < 0]
    dvol = float(downside.std(ddof=1)) * np.sqrt(ppy) if len(downside) > 1 else float("nan")
    sortino = (mean_ann - rf) / dvol if dvol and dvol > 0 else float("nan")
    dd = equity / equity.cummax() - 1.0
    mdd = float(dd.min())
    calmar = cagr / abs(mdd) if mdd < -1e-12 else float("nan")
    active = returns[returns != 0.0]
    hit = float((active > 0).mean()) if len(active) else float("nan")
    return {
        "total_return": total - 1.0,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "hit_rate": hit,
        "n_bars": int(len(returns)),
        "years": years,
    }


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series                 # growth of 1 unit
    equity_usd: pd.Series             # capital * equity
    returns: pd.Series
    positions: pd.Series              # executed positions (already lagged, levered)
    drawdown: pd.Series
    benchmark_equity: pd.Series
    benchmark_returns: pd.Series
    metrics: dict = field(default_factory=dict)
    benchmark_metrics: dict = field(default_factory=dict)
    turnover_annual: float = 0.0
    exposure: float = 0.0
    cost_bps: float = 0.0
    capital: float = 10_000.0
    pnl_usd: float = 0.0


def run_backtest(close: pd.Series, target_position: pd.Series, cost_bps: float = 5.0,
                 rf: float = 0.0, periods_per_year: float = TRADING_DAYS,
                 capital: float = 10_000.0, leverage: float = 1.0) -> BacktestResult:
    """Backtest a target-exposure series against buy & hold.

    `target_position` is the desired exposure decided at the close of bar t
    (using only information up to t); it earns the return of bar t+1.
    `leverage` scales the whole exposure (lot sizing); costs scale with it.
    """
    if capital <= 0 or leverage <= 0:
        raise ValueError("capital and leverage must be positive")
    close = close.dropna()
    pos_target = target_position.reindex(close.index).fillna(0.0) * leverage
    ret = close.pct_change().fillna(0.0)

    pos = pos_target.shift(1).fillna(0.0)           # one-bar execution lag
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])
    costs = turnover * cost_bps * 1e-4
    strat_ret = pos * ret - costs

    equity = (1.0 + strat_ret).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    bench_ret = ret.copy()
    bench_equity = (1.0 + bench_ret).cumprod()

    return BacktestResult(
        equity=equity,
        equity_usd=capital * equity,
        returns=strat_ret,
        positions=pos,
        drawdown=drawdown,
        benchmark_equity=bench_equity,
        benchmark_returns=bench_ret,
        metrics=perf_metrics(strat_ret, rf, periods_per_year),
        benchmark_metrics=perf_metrics(bench_ret, rf, periods_per_year),
        turnover_annual=float(turnover.mean()) * periods_per_year,
        exposure=float((pos.abs() > 1e-12).mean()),
        cost_bps=cost_bps,
        capital=capital,
        pnl_usd=capital * (float(equity.iloc[-1]) - 1.0),
    )


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def moving_average(close: pd.Series, window: int, ma_type: str = "SMA") -> pd.Series:
    if ma_type == "SMA":
        return close.rolling(window).mean()
    if ma_type == "EMA":
        # min_periods=window so the warm-up region stays NaN (flat position).
        return close.ewm(span=window, adjust=False, min_periods=window).mean()
    raise ValueError("ma_type must be 'SMA' or 'EMA'")


# ---------------------------------------------------------------------------
# Signal generators (all causal: information up to and including bar t only)
# ---------------------------------------------------------------------------

def signal_buy_hold(close: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=close.index)


def signal_ma_cross(close: pd.Series, fast: int = 20, slow: int = 50,
                    long_short: bool = False, ma_type: str = "SMA") -> pd.Series:
    if fast >= slow:
        raise ValueError("fast window must be < slow window")
    f = moving_average(close, fast, ma_type)
    s = moving_average(close, slow, ma_type)
    up = (f > s).astype(float)
    pos = up * 2.0 - 1.0 if long_short else up
    pos[f.isna() | s.isna()] = 0.0
    return pos


# Backward-compatible alias (SMA variant).
def signal_sma_cross(close: pd.Series, fast: int = 20, slow: int = 50,
                     long_short: bool = False) -> pd.Series:
    return signal_ma_cross(close, fast, slow, long_short, "SMA")


def signal_kalman_trend(close: pd.Series, resp: float = -3.0,
                        long_short: bool = True) -> pd.Series:
    """Trend following on the *filtered* (causal) drift state of a local
    linear trend model on log-price. The RTS smoother is deliberately not
    used here: it is two-sided and would leak future information.
    """
    y = np.log(close.to_numpy(dtype=float))
    r_hat = max(0.5 * float(np.var(np.diff(y))), 1e-12)
    q_hat = r_hat * 10.0 ** resp
    res = core.kalman(y, model="local_trend", q=q_hat, r=r_hat, q_drift=q_hat * 1e-3)
    drift = np.asarray(res["x_filt"])[:, 1]
    pos = np.sign(drift)
    if not long_short:
        pos = np.maximum(pos, 0.0)
    pos[:20] = 0.0  # filter burn-in
    return pd.Series(pos, index=close.index)


def signal_zscore_meanrev(close: pd.Series, window: int = 20, entry_z: float = 1.5,
                          long_short: bool = True) -> pd.Series:
    """Contrarian: fade |z| > entry_z deviations from the rolling mean."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=1)
    z = (close - mean) / std.replace(0.0, np.nan)
    pos = pd.Series(0.0, index=close.index)
    pos[z < -entry_z] = 1.0
    if long_short:
        pos[z > entry_z] = -1.0
    pos[z.isna()] = 0.0
    return pos


def signal_vol_target(close: pd.Series, target_vol: float = 0.15, window: int = 21,
                      max_leverage: float = 2.0,
                      periods_per_year: float = TRADING_DAYS) -> pd.Series:
    """Constant-risk long exposure: leverage = target / realized volatility."""
    lr = np.log(close / close.shift(1))
    realized = lr.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    lev = (target_vol / realized).clip(0.0, max_leverage)
    return lev.fillna(0.0)


STRATEGIES = (
    "Kalman trend (drift sign)",
    "MA crossover",
    "Mean reversion (z-score)",
    "Volatility targeting",
    "Buy & hold",
)


# ---------------------------------------------------------------------------
# Composable position filters (causal)
# ---------------------------------------------------------------------------

def filter_trend(pos: pd.Series, close: pd.Series, window: int = 200,
                 ma_type: str = "SMA") -> pd.Series:
    """Regime filter: keep longs only above the trend MA, shorts only below."""
    ma = moving_average(close, window, ma_type)
    above = close > ma
    out = pos.copy()
    out[(pos > 0) & ~above] = 0.0
    out[(pos < 0) & above] = 0.0
    out[ma.isna()] = 0.0
    return out


def filter_volume(pos: pd.Series, volume: pd.Series, window: int = 20,
                  mult: float = 1.0) -> pd.Series:
    """Participation filter: hold a position only when current volume exceeds
    `mult` × its rolling average (confirmation of activity)."""
    volume = volume.reindex(pos.index).fillna(0.0)
    avg = volume.rolling(window).mean()
    ok = volume >= mult * avg
    out = pos.copy()
    out[~ok] = 0.0
    out[avg.isna()] = 0.0
    return out


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Year x month compounded return matrix for the calendar heatmap."""
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    frame = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    table = frame.pivot(index="year", columns="month", values="ret")
    return table.reindex(columns=range(1, 13))


def rolling_sharpe(returns: pd.Series, window: int = 126, rf: float = 0.0,
                   periods_per_year: float = TRADING_DAYS) -> pd.Series:
    mean = returns.rolling(window).mean() * periods_per_year - rf
    vol = returns.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    return mean / vol.replace(0.0, np.nan)
