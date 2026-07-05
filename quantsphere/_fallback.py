"""Pure-NumPy mirror of the native qsengine module.

Same function signatures, same risk-neutral dynamics, same validation rules.
Used automatically when the compiled extension is unavailable (e.g. Streamlit
Community Cloud without a build step).
"""

from __future__ import annotations

import os

import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.sparse.linalg import splu
from scipy.stats import norm

_CHUNK = 20_000


def version() -> str:
    return "1.0.0"


def hardware_threads() -> int:
    return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def _validate(model: str, S0, T, sigma, v0, kappa, theta, xi, rho, vg_theta, vg_nu) -> None:
    if S0 <= 0 or T <= 0:
        raise ValueError("S0 and T must be positive")
    if model in ("gbm", "merton", "vg") and sigma <= 0:
        raise ValueError("sigma must be positive")
    if model == "heston":
        if v0 < 0 or theta < 0 or kappa < 0 or xi < 0:
            raise ValueError("Heston parameters must be non-negative")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [-1,1]")
    if model == "vg":
        if vg_nu <= 0:
            raise ValueError("vg_nu must be positive")
        if 1.0 - vg_theta * vg_nu - 0.5 * sigma * sigma * vg_nu <= 0:
            raise ValueError(
                "VG martingale correction undefined: require 1 - theta*nu - 0.5*sigma^2*nu > 0"
            )


def _simulate_block(model, rng, n, steps, dt, S0, r, sigma, v0, kappa, theta, xi, rho,
                    lam, mu_j, sig_j, vg_theta, vg_nu, keep_paths, barrier_log, barrier_up):
    """Simulate `n` log-price paths stepwise.

    Returns (paths | None, terminal_log, touched | None). Memory stays O(n)
    when keep_paths is False.
    """
    sqdt = np.sqrt(dt)
    x = np.full(n, np.log(S0))
    paths = np.empty((n, steps + 1)) if keep_paths else None
    if keep_paths:
        paths[:, 0] = x
    touched = None
    if barrier_log is not None:
        touched = (x >= barrier_log) if barrier_up else (x <= barrier_log)

    if model == "heston":
        v = np.full(n, v0)
        srho = np.sqrt(1.0 - rho * rho)
    if model == "merton":
        k_comp = np.exp(mu_j + 0.5 * sig_j**2) - 1.0
    if model == "vg":
        omega = np.log(1.0 - vg_theta * vg_nu - 0.5 * sigma * sigma * vg_nu) / vg_nu

    for i in range(1, steps + 1):
        if model == "gbm":
            x = x + (r - 0.5 * sigma * sigma) * dt + sigma * sqdt * rng.standard_normal(n)
        elif model == "heston":
            vp = np.maximum(v, 0.0)
            z1 = rng.standard_normal(n)
            z2 = rho * z1 + srho * rng.standard_normal(n)
            x = x + (r - 0.5 * vp) * dt + np.sqrt(vp) * sqdt * z1
            v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sqdt * z2
        elif model == "merton":
            nj = rng.poisson(lam * dt, size=n)
            jump = nj * mu_j + sig_j * np.sqrt(nj) * rng.standard_normal(n)
            x = (x + (r - 0.5 * sigma * sigma - lam * k_comp) * dt
                 + sigma * sqdt * rng.standard_normal(n) + jump)
        elif model == "vg":
            g = rng.gamma(dt / vg_nu, vg_nu, size=n)
            x = x + (r + omega) * dt + vg_theta * g + sigma * np.sqrt(g) * rng.standard_normal(n)
        else:
            raise ValueError(f"unknown model: {model}")
        if keep_paths:
            paths[:, i] = x
        if touched is not None:
            touched |= (x >= barrier_log) if barrier_up else (x <= barrier_log)
    return paths, x, touched


def simulate_paths(model, n_paths, steps, seed=42, S0=100.0, r=0.02, T=1.0, sigma=0.2,
                   v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, lam=0.5,
                   mu_j=-0.05, sig_j=0.15, vg_theta=-0.1, vg_nu=0.2):
    _validate(model, S0, T, sigma, v0, kappa, theta, xi, rho, vg_theta, vg_nu)
    if steps < 1 or n_paths < 1:
        raise ValueError("steps and n_paths must be >= 1")
    rng = np.random.default_rng(seed)
    paths, _, _ = _simulate_block(model, rng, int(n_paths), int(steps), T / steps, S0, r,
                                  sigma, v0, kappa, theta, xi, rho, lam, mu_j, sig_j,
                                  vg_theta, vg_nu, True, None, True)
    return np.exp(paths)


def mc_price(model, K, is_call, n_paths=50_000, steps=252, seed=42, barrier_type="none",
             barrier=0.0, S0=100.0, r=0.02, T=1.0, sigma=0.2, v0=0.04, kappa=1.5,
             theta=0.04, xi=0.5, rho=-0.7, lam=0.5, mu_j=-0.05, sig_j=0.15,
             vg_theta=-0.1, vg_nu=0.2):
    _validate(model, S0, T, sigma, v0, kappa, theta, xi, rho, vg_theta, vg_nu)
    if K <= 0:
        raise ValueError("strike must be positive")
    if steps < 1 or n_paths < 1:
        raise ValueError("steps and n_paths must be >= 1")
    if barrier_type not in ("none", "up-and-out", "down-and-out", "up-and-in", "down-and-in"):
        raise ValueError(f"unknown barrier type: {barrier_type}")
    if barrier_type != "none" and barrier <= 0:
        raise ValueError("barrier level must be positive")

    n_paths = int(n_paths)
    rng = np.random.default_rng(seed)
    barrier_log = np.log(barrier) if barrier_type != "none" else None
    barrier_up = barrier_type in ("up-and-out", "up-and-in")
    knock_in = barrier_type in ("up-and-in", "down-and-in")

    terminal = np.empty(n_paths)
    total, total2 = 0.0, 0.0
    done = 0
    while done < n_paths:
        n = min(_CHUNK, n_paths - done)
        _, x_end, touched = _simulate_block(model, rng, n, int(steps), T / steps, S0, r,
                                            sigma, v0, kappa, theta, xi, rho, lam, mu_j,
                                            sig_j, vg_theta, vg_nu, False, barrier_log,
                                            barrier_up)
        ST = np.exp(x_end)
        terminal[done:done + n] = ST
        payoff = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
        if touched is not None:
            alive = touched if knock_in else ~touched
            payoff = payoff * alive
        total += float(payoff.sum())
        total2 += float((payoff * payoff).sum())
        done += n

    disc = np.exp(-r * T)
    mean = total / n_paths
    var = max(0.0, total2 / n_paths - mean * mean)
    return {
        "price": disc * mean,
        "std_error": disc * np.sqrt(var / n_paths),
        "terminal": terminal,
    }


# ---------------------------------------------------------------------------
# PDE — Crank-Nicolson (operator-splitting projection for American exercise)
# ---------------------------------------------------------------------------

def pde_price(S0, K, r, sigma, T, is_call=True, american=False, Ns=200, Nt=200,
              s_max_mult=3.0):
    if S0 <= 0 or K <= 0 or T <= 0:
        raise ValueError("S0, K, T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if Ns < 10 or Nt < 10:
        raise ValueError("Ns and Nt must be >= 10")

    smax = max(s_max_mult, np.exp(3.0 * sigma * np.sqrt(T))) * max(S0, K)
    dS = smax / Ns
    dt = T / Nt
    s_grid = np.arange(Ns + 1) * dS
    t_grid = np.arange(Nt + 1) * dt
    payoff = np.maximum(s_grid - K, 0.0) if is_call else np.maximum(K - s_grid, 0.0)

    surface = np.zeros((Nt + 1, Ns + 1))
    surface[Nt] = payoff
    V = payoff.copy()

    i = np.arange(1, Ns)
    A = 0.25 * dt * (sigma**2 * i**2 - r * i)
    B = -0.5 * dt * (sigma**2 * i**2 + r)
    C = 0.25 * dt * (sigma**2 * i**2 + r * i)

    n = Ns - 1
    left = sparse.diags([-A[1:], 1.0 - B, -C[:-1]], [-1, 0, 1], format="csc")
    lu = splu(left)
    psi = payoff[1:-1]

    def lower_bc(tau):
        if is_call:
            return 0.0
        return K if american else K * np.exp(-r * tau)

    def upper_bc(tau):
        return smax - K * np.exp(-r * tau) if is_call else 0.0

    for j in range(Nt - 1, -1, -1):
        tau_new = T - j * dt
        tau_old = T - (j + 1) * dt
        rhs = A * V[:-2] + (1.0 + B) * V[1:-1] + C * V[2:]
        rhs[0] += A[0] * lower_bc(tau_old) + A[0] * lower_bc(tau_new)
        rhs[-1] += C[-1] * upper_bc(tau_old) + C[-1] * upper_bc(tau_new)
        x = lu.solve(rhs)
        if american:
            x = np.maximum(x, psi)
        V[0] = lower_bc(tau_new)
        V[-1] = upper_bc(tau_new)
        V[1:-1] = x
        surface[j] = V

    pos = np.clip(S0 / dS, 0.0, float(Ns))
    idx = min(int(pos), Ns - 1)
    w = pos - idx

    def value_at(layer, S):
        p = np.clip(S / dS, 0.0, float(Ns))
        k = min(int(p), Ns - 1)
        ww = p - k
        return layer[k] * (1.0 - ww) + layer[k + 1] * ww

    price = surface[0][idx] * (1.0 - w) + surface[0][idx + 1] * w
    vu = value_at(surface[0], min(S0 + dS, smax))
    vd = value_at(surface[0], max(S0 - dS, 0.0))
    return {
        "price": float(price),
        "delta": float((vu - vd) / (2.0 * dS)),
        "gamma": float((vu - 2.0 * price + vd) / (dS * dS)),
        "theta": float((value_at(surface[1], S0) - price) / dt),
        "s_grid": s_grid,
        "t_grid": t_grid,
        "surface": surface,
    }


# ---------------------------------------------------------------------------
# Kalman filter + RTS smoother (scalar observation, dim 1 or 2)
# ---------------------------------------------------------------------------

def kalman(y, model="local_level", q=1e-5, r=1e-3, q_drift=1e-8):
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 2:
        raise ValueError("need at least 2 observations")
    if r <= 0:
        raise ValueError("R must be positive")

    if model == "local_level":
        d = 1
        Amat = np.array([[1.0]])
        Cvec = np.array([1.0])
        Qmat = np.array([[q]])
        x = np.array([y[0]])
        P = np.array([[r * 10.0 + 1e-8]])
    elif model == "local_trend":
        d = 2
        Amat = np.array([[1.0, 1.0], [0.0, 1.0]])
        Cvec = np.array([1.0, 0.0])
        Qmat = np.diag([q, q_drift])
        x = np.array([y[0], 0.0])
        P = np.diag([r * 10.0 + 1e-8, r + 1e-8])
    else:
        raise ValueError("model must be 'local_level' or 'local_trend'")

    x_filt = np.zeros((n, d))
    P_filt = np.zeros((n, d, d))
    x_pred = np.zeros((n, d))
    P_pred = np.zeros((n, d, d))
    innovations = np.zeros(n)
    innov_var = np.zeros(n)
    loglik = 0.0
    LOG2PI = np.log(2.0 * np.pi)

    for t in range(n):
        if t == 0:
            xp, Pp = x, P
        else:
            xp = Amat @ x
            Pp = Amat @ P @ Amat.T + Qmat
        x_pred[t], P_pred[t] = xp, Pp

        S = float(Cvec @ Pp @ Cvec + r)
        v = y[t] - float(Cvec @ xp)
        Kg = (Pp @ Cvec) / S
        x = xp + Kg * v
        P = Pp - np.outer(Kg, Cvec @ Pp)
        P = 0.5 * (P + P.T)

        innovations[t] = v
        innov_var[t] = S
        loglik += -0.5 * (LOG2PI + np.log(S) + v * v / S)
        x_filt[t], P_filt[t] = x, P

    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()
    for t in range(n - 2, -1, -1):
        Ppn = P_pred[t + 1] + np.eye(d) * 1e-12
        G = P_filt[t] @ Amat.T @ np.linalg.inv(Ppn)
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred[t + 1]) @ G.T

    return {
        "x_filt": x_filt,
        "P_filt": P_filt.reshape(n, d * d),
        "x_smooth": x_smooth,
        "P_smooth": P_smooth.reshape(n, d * d),
        "innovations": innovations,
        "innov_var": innov_var,
        "loglik": float(loglik),
        "dim": d,
    }


# ---------------------------------------------------------------------------
# Black-Scholes analytics & implied volatility
# ---------------------------------------------------------------------------

def bs_price(S, K, T, r, sigma, is_call=True):
    if S <= 0 or K <= 0:
        return float("nan")
    if T <= 0 or sigma <= 0:
        fwd = S - K * np.exp(-r * T) if is_call else K * np.exp(-r * T) - S
        return float(max(fwd, 0.0))
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    if is_call:
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def implied_vol(prices, S, strikes, maturities, r, is_call):
    prices = np.asarray(prices, dtype=float).ravel()
    strikes = np.asarray(strikes, dtype=float).ravel()
    maturities = np.asarray(maturities, dtype=float).ravel()
    is_call = np.asarray(is_call, dtype=bool).ravel()
    if not (prices.size == strikes.size == maturities.size == is_call.size):
        raise ValueError("all arrays must share the same length")

    out = np.full(prices.size, np.nan)
    lo_v, hi_v = 1e-6, 5.0
    for idx in range(prices.size):
        p, k, t, c = prices[idx], strikes[idx], maturities[idx], bool(is_call[idx])
        if not (p > 0 and k > 0 and t > 0 and S > 0):
            continue
        lower = bs_price(S, k, t, r, lo_v, c)
        upper = bs_price(S, k, t, r, hi_v, c)
        if p <= lower + 1e-12 or p >= upper - 1e-12:
            continue
        try:
            out[idx] = brentq(lambda s: bs_price(S, k, t, r, s, c) - p, lo_v, hi_v,
                              xtol=1e-10, maxiter=200)
        except ValueError:
            pass
    return out
