"""Stage 7 — volatility models and probabilistic price forecasting.

GARCH(1,1) by exact maximum likelihood, RiskMetrics EWMA, analytic GBM
forecast cones, bootstrap cones, and the forward price density surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

TRADING_DAYS = 252

# Returns are scaled to percent inside the GARCH likelihood for conditioning;
# all public outputs are converted back to decimal/annualized units.
_SCALE = 100.0


# ---------------------------------------------------------------------------
# GARCH(1,1):  r_t = sigma_t eps_t,  sigma^2_t = omega + alpha r^2_{t-1} + beta sigma^2_{t-1}
# ---------------------------------------------------------------------------

def _garch_filter(r2: np.ndarray, omega: float, alpha: float, beta: float,
                  s2_0: float) -> np.ndarray:
    n = r2.size
    s2 = np.empty(n)
    s2[0] = s2_0
    for t in range(1, n):
        s2[t] = omega + alpha * r2[t - 1] + beta * s2[t - 1]
    return s2


def fit_garch11(returns: pd.Series | np.ndarray) -> dict:
    """MLE fit of a Gaussian GARCH(1,1) on daily log-returns (decimal units).

    Returns dict with omega/alpha/beta (percent^2 scale internally converted),
    the conditional annualized vol series, persistence, half-life, and both
    the unconditional and latest annualized volatilities.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 250:
        raise ValueError("GARCH needs at least 250 return observations")
    rp = (r - r.mean()) * _SCALE
    r2 = rp**2
    var_u = float(rp.var())

    def neg_loglik(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9995:
            return 1e10
        s2 = _garch_filter(r2, omega, alpha, beta, var_u)
        if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
            return 1e10
        return 0.5 * float(np.sum(np.log(s2) + r2 / s2))

    best = None
    for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.02, 0.95)):
        x0 = np.array([var_u * (1.0 - a0 - b0), a0, b0])
        res = minimize(neg_loglik, x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, beta = best.x
    if not (omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta < 1.0):
        raise RuntimeError("GARCH MLE did not converge to a stationary solution")

    s2 = _garch_filter(r2, omega, alpha, beta, var_u)
    persistence = alpha + beta
    uncond_var = omega / (1.0 - persistence)
    idx = returns.index if isinstance(returns, pd.Series) else pd.RangeIndex(r.size)
    cond_vol_ann = pd.Series(np.sqrt(s2) / _SCALE * np.sqrt(TRADING_DAYS), index=idx[-r.size:])
    return {
        "omega": float(omega), "alpha": float(alpha), "beta": float(beta),
        "persistence": float(persistence),
        "half_life": float(np.log(0.5) / np.log(persistence)) if persistence < 1 else float("inf"),
        "loglik": float(-best.fun),
        "cond_vol_annual": cond_vol_ann,
        "uncond_vol_annual": float(np.sqrt(uncond_var) / _SCALE * np.sqrt(TRADING_DAYS)),
        "last_vol_annual": float(cond_vol_ann.iloc[-1]),
        "last_var_daily_pct": float(s2[-1]),
        "last_r2_pct": float(r2[-1]),
    }


def garch_forecast_vol(fit: dict, horizon_days: int) -> np.ndarray:
    """Term structure E[sigma^2_{t+h}] annualized, h = 1..horizon.

    E[s2_{t+h}] = s2_bar + (alpha+beta)^{h-1} (s2_{t+1} - s2_bar).
    """
    omega, alpha, beta = fit["omega"], fit["alpha"], fit["beta"]
    pers = fit["persistence"]
    s2_bar = omega / (1.0 - pers)
    s2_next = omega + alpha * fit["last_r2_pct"] + beta * fit["last_var_daily_pct"]
    h = np.arange(1, horizon_days + 1)
    s2_h = s2_bar + pers ** (h - 1) * (s2_next - s2_bar)
    return np.sqrt(s2_h) / _SCALE * np.sqrt(TRADING_DAYS)


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA conditional volatility (annualized)."""
    r = returns.dropna().to_numpy(dtype=float)
    if r.size < 30:
        raise ValueError("EWMA needs at least 30 observations")
    s2 = np.empty(r.size)
    s2[0] = float(np.var(r[:30]))
    for t in range(1, r.size):
        s2[t] = lam * s2[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return pd.Series(np.sqrt(s2 * TRADING_DAYS), index=returns.dropna().index)


# ---------------------------------------------------------------------------
# Forecast cones
# ---------------------------------------------------------------------------

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def cone_gbm(S0: float, mu: float, sigma: float, days: int,
             quantiles: tuple[float, ...] = QUANTILES) -> dict:
    """Analytic lognormal forecast cone under GBM with drift mu (real-world).

    S_t = S0 exp((mu - sigma^2/2) t + sigma sqrt(t) Z_q).
    """
    if S0 <= 0 or sigma <= 0 or days < 1:
        raise ValueError("S0, sigma must be positive and days >= 1")
    t = np.arange(1, days + 1) / TRADING_DAYS
    out = {"t_days": np.arange(1, days + 1)}
    for q in quantiles:
        z = norm.ppf(q)
        out[q] = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)
    out["mean"] = S0 * np.exp(mu * t)
    return out


def cone_bootstrap(S0: float, log_returns: pd.Series | np.ndarray, days: int,
                   n_sims: int = 10_000, seed: int = 21, block: int = 5,
                   quantiles: tuple[float, ...] = QUANTILES) -> dict:
    """Circular block-bootstrap forecast cone (model-free, keeps fat tails
    and short-range autocorrelation up to the block length)."""
    lr = np.asarray(log_returns, dtype=float)
    lr = lr[np.isfinite(lr)]
    if lr.size < 100:
        raise ValueError("Bootstrap needs at least 100 return observations")
    if S0 <= 0 or days < 1 or n_sims < 100:
        raise ValueError("invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(days / block))
    starts = rng.integers(0, lr.size, size=(n_sims, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % lr.size
    draws = lr[idx].reshape(n_sims, n_blocks * block)[:, :days]
    paths = S0 * np.exp(np.cumsum(draws, axis=1))
    out = {"t_days": np.arange(1, days + 1)}
    for q in quantiles:
        out[q] = np.quantile(paths, q, axis=0)
    out["mean"] = paths.mean(axis=0)
    out["terminal"] = paths[:, -1]
    return out


def density_surface_gbm(S0: float, mu: float, sigma: float, days: int,
                        n_t: int = 36, n_s: int = 90) -> dict:
    """Forward transition density p(S, t) under GBM (lognormal), for the 3-D
    probability landscape: x = horizon, y = price, z = density."""
    if S0 <= 0 or sigma <= 0:
        raise ValueError("S0 and sigma must be positive")
    t = np.linspace(5.0 / TRADING_DAYS, days / TRADING_DAYS, n_t)
    smax = S0 * np.exp((mu - 0.5 * sigma**2) * t[-1] + 2.8 * sigma * np.sqrt(t[-1]))
    smin = S0 * np.exp((mu - 0.5 * sigma**2) * t[-1] - 2.8 * sigma * np.sqrt(t[-1]))
    s = np.linspace(smin, smax, n_s)
    T, S = np.meshgrid(t, s)
    m = (mu - 0.5 * sigma**2) * T
    v = sigma * np.sqrt(T)
    Z = np.exp(-0.5 * ((np.log(S / S0) - m) / v) ** 2) / (S * v * np.sqrt(2.0 * np.pi))
    return {"t_years": t, "s_grid": s, "density": Z}


def prob_above(S0: float, target: float, mu: float, sigma: float, t_years: float) -> float:
    """P(S_t > target) under GBM with real-world drift mu."""
    if S0 <= 0 or target <= 0 or sigma <= 0 or t_years <= 0:
        raise ValueError("all inputs must be positive")
    d = (np.log(target / S0) - (mu - 0.5 * sigma**2) * t_years) / (sigma * np.sqrt(t_years))
    return float(1.0 - norm.cdf(d))


def horizon_var_es(terminal: np.ndarray, S0: float, alpha: float = 0.05) -> tuple[float, float]:
    """Horizon VaR and expected shortfall (losses as positive fractions)."""
    rets = terminal / S0 - 1.0
    q = float(np.quantile(rets, alpha))
    tail = rets[rets <= q]
    es = float(tail.mean()) if tail.size else q
    return -q, -es
