"""Stage 5 — Markowitz mean-variance optimization on a real-ticker basket."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quantsphere.data import yf_download

TRADING_DAYS = 252


def fetch_basket(tickers: tuple[str, ...], years: int = 5) -> pd.DataFrame:
    """Aligned adjusted close matrix for the basket (inner-joined calendar)."""
    if len(tickers) < 2:
        raise ValueError("Need at least 2 tickers for portfolio optimization")
    df = yf_download(list(tickers), period=f"{years}y", interval="1d", auto_adjust=True)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    close = close.dropna(axis=1, how="all").dropna()
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise ValueError(f"No data for: {', '.join(missing)}")
    if len(close) < 120:
        raise ValueError("Insufficient overlapping history across the basket")
    return close[list(tickers)]


def annual_stats(prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Annualized mean vector and covariance matrix from daily log-returns.

    The covariance is ridge-regularized until positive definite, so a
    near-singular basket (e.g. duplicated tickers) cannot break the solvers.
    """
    lr = np.log(prices / prices.shift(1)).dropna()
    mu = lr.mean().to_numpy() * TRADING_DAYS
    cov = lr.cov().to_numpy() * TRADING_DAYS

    jitter = 0.0
    base = np.trace(cov) / len(cov)
    for _ in range(12):
        try:
            np.linalg.cholesky(cov + jitter * np.eye(len(cov)))
            break
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10.0, base * 1e-10)
    cov = cov + jitter * np.eye(len(cov))
    return mu, cov


def portfolio_perf(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 1e-18)))
    return ret, vol


def random_portfolios(mu: np.ndarray, cov: np.ndarray, n: int = 4000, rf: float = 0.02,
                      seed: int = 7) -> dict[str, np.ndarray]:
    """Dirichlet-sampled long-only portfolios for the frontier cloud."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(mu)), size=n)
    rets = W @ mu
    vols = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", W, cov, W), 1e-18))
    sharpes = (rets - rf) / vols
    return {"weights": W, "returns": rets, "vols": vols, "sharpes": sharpes}


def _solve(objective, n_assets: int, extra_constraints=()) -> np.ndarray:
    w0 = np.full(n_assets, 1.0 / n_assets)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, *extra_constraints]
    res = minimize(objective, w0, method="SLSQP", bounds=[(0.0, 1.0)] * n_assets,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"Optimizer failed: {res.message}")
    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum()


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.02) -> np.ndarray:
    def neg_sharpe(w):
        ret, vol = portfolio_perf(w, mu, cov)
        return -(ret - rf) / vol

    return _solve(neg_sharpe, len(mu))


def min_variance(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return _solve(lambda w: float(w @ cov @ w), len(mu))


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 40) -> dict[str, np.ndarray]:
    """Long-only efficient frontier between the min-var return and max asset return."""
    w_minvar = min_variance(mu, cov)
    ret_lo, _ = portfolio_perf(w_minvar, mu, cov)
    ret_hi = float(mu.max())
    targets = np.linspace(ret_lo, ret_hi, n_points)

    vols, rets = [], []
    for target in targets:
        try:
            w = _solve(lambda w: float(w @ cov @ w), len(mu),
                       extra_constraints=[{"type": "eq",
                                           "fun": lambda w, t=target: float(w @ mu) - t}])
        except RuntimeError:
            continue
        ret, vol = portfolio_perf(w, mu, cov)
        rets.append(ret)
        vols.append(vol)
    return {"returns": np.array(rets), "vols": np.array(vols)}
