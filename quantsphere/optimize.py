"""Stage 8 — hyperparameter optimization with honest walk-forward validation.

The methodology that separates research from curve-fitting:
- every candidate is backtested causally (1-bar lag, causal signals);
- the sample is split chronologically: parameters are *ranked on the train
  segment's out-of-sample continuation* (the test segment), and both scores
  are reported so overfitting is visible, not hidden.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from quantsphere import backtest as bt

OPTIMIZABLE = (
    "MA crossover",
    "Kalman trend (drift sign)",
    "Mean reversion (z-score)",
    "Volatility targeting",
)

# Human-readable axis names per strategy: (param_1, param_2) — param_2 None
# for one-dimensional searches.
PARAM_AXES = {
    "MA crossover": ("fast", "slow"),
    "Kalman trend (drift sign)": ("resp", None),
    "Mean reversion (z-score)": ("window", "entry_z"),
    "Volatility targeting": ("window", "target_vol"),
}


def param_grid(strategy: str) -> list[dict]:
    if strategy == "MA crossover":
        fasts = [5, 8, 10, 15, 20, 25, 30, 40, 50]
        slows = [20, 30, 40, 60, 80, 100, 130, 160, 200, 250]
        return [{"fast": f, "slow": s, "ma_type": m}
                for m in ("SMA", "EMA")
                for f in fasts for s in slows if f < s]
    if strategy == "Kalman trend (drift sign)":
        return [{"resp": float(r)} for r in np.round(np.linspace(-6.0, -1.0, 26), 2)]
    if strategy == "Mean reversion (z-score)":
        wins = [5, 8, 10, 15, 20, 25, 30, 40, 60]
        entries = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
        return [{"window": w, "entry_z": e} for w in wins for e in entries]
    if strategy == "Volatility targeting":
        targets = np.round(np.arange(0.05, 0.45, 0.05), 2)
        wins = [10, 15, 21, 30, 42, 63]
        return [{"window": w, "target_vol": float(t)} for w in wins for t in targets]
    raise ValueError(f"'{strategy}' is not optimizable")


def build_signal(close: pd.Series, strategy: str, params: dict, long_short: bool,
                 ppy: float) -> pd.Series:
    if strategy == "MA crossover":
        return bt.signal_ma_cross(close, params["fast"], params["slow"],
                                  long_short, params.get("ma_type", "SMA"))
    if strategy == "Kalman trend (drift sign)":
        return bt.signal_kalman_trend(close, params["resp"], long_short)
    if strategy == "Mean reversion (z-score)":
        return bt.signal_zscore_meanrev(close, params["window"], params["entry_z"],
                                        long_short)
    if strategy == "Volatility targeting":
        return bt.signal_vol_target(close, params["target_vol"], params["window"],
                                    periods_per_year=ppy)
    raise ValueError(f"unknown strategy '{strategy}'")


def grid_search(close: pd.Series, strategy: str, cost_bps: float = 5.0, rf: float = 0.0,
                ppy: float = 252.0, long_short: bool = False, train_frac: float = 0.7,
                progress: Callable[[float], None] | None = None) -> pd.DataFrame:
    """Exhaustive causal grid search with a chronological train/test split.

    Signals are causal, so computing them on the full series then scoring the
    two segments separately leaks nothing: the value of a signal at bar t
    never depends on data after t (verified by the causality test suite).
    """
    if not 0.5 <= train_frac <= 0.95:
        raise ValueError("train_frac must be in [0.5, 0.95]")
    close = close.dropna()
    grid = param_grid(strategy)
    n = len(close)
    cut = int(n * train_frac)
    if cut < 60 or n - cut < 40:
        raise ValueError("Not enough bars for a train/test split")
    split_date = close.index[cut]

    rows = []
    for i, params in enumerate(grid):
        try:
            sig = build_signal(close, strategy, params, long_short, ppy)
            res = bt.run_backtest(close, sig, cost_bps=cost_bps, rf=rf,
                                  periods_per_year=ppy)
            m_tr = bt.perf_metrics(res.returns.iloc[:cut], rf, ppy)
            m_te = bt.perf_metrics(res.returns.iloc[cut:], rf, ppy)
        except (ValueError, RuntimeError):
            continue
        rows.append({
            **params,
            "sharpe_train": m_tr["sharpe"], "sharpe_test": m_te["sharpe"],
            "cagr_train": m_tr["cagr"], "cagr_test": m_te["cagr"],
            "mdd_test": m_te["max_drawdown"], "calmar_test": m_te["calmar"],
            "total_test": m_te["total_return"],
        })
        if progress is not None:
            progress((i + 1) / len(grid))

    if not rows:
        raise RuntimeError("No parameter combination produced a valid backtest")
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["sharpe_test", "sharpe_train"])
    df.attrs["split_date"] = split_date
    df.attrs["n_bars"] = n
    df.attrs["n_combos"] = len(grid)
    return df.sort_values("sharpe_test", ascending=False).reset_index(drop=True)


def surface_pivot(df: pd.DataFrame, strategy: str,
                  value: str = "sharpe_test") -> pd.DataFrame | None:
    """Pivot the results into a (param_1 x param_2) matrix for 3-D/heatmap
    rendering. Returns None for one-dimensional searches."""
    p1, p2 = PARAM_AXES[strategy]
    if p2 is None:
        return None
    sub = df
    if strategy == "MA crossover":
        # Surface over the dominant MA type among the top results.
        top_type = df.iloc[0].get("ma_type", "SMA")
        sub = df[df["ma_type"] == top_type]
    return sub.pivot_table(index=p2, columns=p1, values=value, aggfunc="mean")
