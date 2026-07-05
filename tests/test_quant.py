"""Verification of the backtest engine and forecasting models — network-free.

Run:  python tests/test_quant.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantsphere import backtest as bt  # noqa: E402
from quantsphere import models as qm  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


idx = pd.bdate_range("2018-01-02", periods=1500)

# --- No look-ahead: a signal that "knows" a jump cannot capture it ------------
prices = pd.Series(100.0, index=idx)
prices.iloc[1000:] = 150.0  # +50% jump at bar 1000
clairvoyant = pd.Series(0.0, index=idx)
clairvoyant.iloc[1000] = 1.0  # turns long exactly on the jump bar
res = bt.run_backtest(prices, clairvoyant, cost_bps=0.0)
check("look-ahead guard: jump bar return is zero",
      abs(res.returns.iloc[1000]) < 1e-15, f"r={res.returns.iloc[1000]:.2e}")
check("position lagged by one bar", res.positions.iloc[1000] == 0.0
      and res.positions.iloc[1001] == 1.0)

# --- Buy & hold reproduces the price ratio (no costs) --------------------------
rng = np.random.default_rng(4)
close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(idx)))), index=idx)
bh = bt.run_backtest(close, bt.signal_buy_hold(close), cost_bps=0.0)
ratio = float(close.iloc[-1] / close.iloc[0])
check("buy & hold equity == price ratio", abs(float(bh.equity.iloc[-1]) - ratio) < 1e-9,
      f"{bh.equity.iloc[-1]:.6f} vs {ratio:.6f}")

# --- Costs strictly reduce performance -----------------------------------------
sig = bt.signal_sma_cross(close, 10, 50)
r0 = bt.run_backtest(close, sig, cost_bps=0.0)
r25 = bt.run_backtest(close, sig, cost_bps=25.0)
check("costs reduce final equity",
      float(r25.equity.iloc[-1]) < float(r0.equity.iloc[-1]),
      f"{r25.equity.iloc[-1]:.4f} < {r0.equity.iloc[-1]:.4f}")

# --- Metrics sanity --------------------------------------------------------------
const = pd.Series(0.0005, index=idx)
mm = bt.perf_metrics(const)
check("constant gains: zero drawdown", mm["max_drawdown"] == 0.0)
check("constant gains: positive sharpe", mm["sharpe"] > 5)
check("CAGR consistency", abs(mm["cagr"] - ((1.0005) ** 252 - 1)) < 1e-9)

# --- Signals are causal (recompute on truncated data gives same past values) ----
for name, fn in (("sma", lambda c: bt.signal_sma_cross(c, 10, 30)),
                 ("zscore", lambda c: bt.signal_zscore_meanrev(c, 20, 1.5)),
                 ("voltarget", lambda c: bt.signal_vol_target(c, 0.15, 21)),
                 ("kalman", lambda c: bt.signal_kalman_trend(c, -3.0))):
    full = fn(close)
    trunc = fn(close.iloc[:1200])
    gap = float(np.nanmax(np.abs(full.iloc[:1200].to_numpy() - trunc.to_numpy())))
    check(f"causality: {name} unchanged by future data", gap < 1e-10, f"max gap={gap:.2e}")

# --- GARCH(1,1): parameter recovery on simulated data ---------------------------
w_true, a_true, b_true = 0.04, 0.08, 0.90  # percent^2 scale
n = 4000
rng = np.random.default_rng(11)
r_pct = np.empty(n)
s2 = w_true / (1 - a_true - b_true)
for t in range(n):
    r_pct[t] = np.sqrt(s2) * rng.standard_normal()
    s2 = w_true + a_true * r_pct[t] ** 2 + b_true * s2
fit = qm.fit_garch11(pd.Series(r_pct / 100.0))
check("GARCH persistence recovered",
      abs(fit["persistence"] - (a_true + b_true)) < 0.05,
      f"{fit['persistence']:.3f} vs {a_true + b_true:.3f}")
check("GARCH alpha in range", abs(fit["alpha"] - a_true) < 0.05, f"{fit['alpha']:.3f}")
check("GARCH stationary", fit["persistence"] < 1.0)

fc = qm.garch_forecast_vol(fit, 500)
uncond = fit["uncond_vol_annual"]
check("GARCH forecast converges to unconditional vol",
      abs(fc[-1] - uncond) < 0.02 * uncond + 1e-9,
      f"{fc[-1]:.4f} → {uncond:.4f}")
mono = np.all(np.diff(np.abs(fc - uncond)) <= 1e-12)
check("GARCH forecast monotone toward σ̄", bool(mono))

# --- EWMA ------------------------------------------------------------------------
ew = qm.ewma_vol(pd.Series(r_pct / 100.0))
check("EWMA positive & finite", bool(np.all(np.isfinite(ew)) and np.all(ew > 0)))

# --- Forecast cones ----------------------------------------------------------------
cone = qm.cone_gbm(100.0, 0.08, 0.25, 252)
check("GBM cone quantiles ordered",
      bool(np.all(cone[0.05] < cone[0.25]) and np.all(cone[0.25] < cone[0.50])
           and np.all(cone[0.50] < cone[0.75]) and np.all(cone[0.75] < cone[0.95])))
med_expected = 100.0 * np.exp((0.08 - 0.5 * 0.25**2) * 1.0)
check("GBM cone median analytic", abs(cone[0.50][-1] - med_expected) < 1e-9,
      f"{cone[0.50][-1]:.4f} vs {med_expected:.4f}")
check("GBM cone mean = S0 e^{mu t}", abs(cone["mean"][-1] - 100 * np.exp(0.08)) < 1e-9)

lr = pd.Series(rng.normal(0.0003, 0.012, 2000))
cb = qm.cone_bootstrap(100.0, lr, 126, n_sims=4000, seed=2)
check("bootstrap cone quantiles ordered",
      bool(np.all(cb[0.05] <= cb[0.50]) and np.all(cb[0.50] <= cb[0.95])))
check("bootstrap terminal size", cb["terminal"].shape == (4000,))

p = qm.prob_above(100.0, 100.0, 0.5 * 0.25**2, 0.25, 1.0)
check("prob_above symmetric case = 50%", abs(p - 0.5) < 1e-12, f"{p:.4f}")
var_h, es_h = qm.horizon_var_es(cb["terminal"], 100.0, 0.05)
check("ES ≥ VaR", es_h >= var_h - 1e-12, f"VaR={var_h:.4f} ES={es_h:.4f}")

dens = qm.density_surface_gbm(100.0, 0.05, 0.2, 252)
z = dens["density"]
check("density surface finite & non-negative",
      bool(np.all(np.isfinite(z)) and np.all(z >= 0)))
ds = dens["s_grid"][1] - dens["s_grid"][0]
mass = float(z[:, -1].sum() * ds)
check("density integrates to ≈ 1 at horizon", abs(mass - 1.0) < 0.05, f"mass={mass:.3f}")

# --- Short positions, EMA, filters, lot sizing, timeframe annualization -------------
down = pd.Series(np.linspace(200.0, 100.0, 400), index=idx[:400])
sig_ls = bt.signal_ma_cross(down, 10, 30, long_short=True)
check("shorts: MA cross emits -1 in a downtrend", float(sig_ls.min()) == -1.0
      and (sig_ls == -1.0).sum() > 100, f"{int((sig_ls == -1).sum())} short bars")
sig_lf = bt.signal_ma_cross(down, 10, 30, long_short=False)
check("long-flat never short", float(sig_lf.min()) == 0.0)
res_ls = bt.run_backtest(down, sig_ls, cost_bps=0.0)
check("shorting a downtrend is profitable", float(res_ls.equity.iloc[-1]) > 1.5,
      f"equity={res_ls.equity.iloc[-1]:.3f}")

ema_full = bt.signal_ma_cross(close, 10, 30, ma_type="EMA")
ema_trunc = bt.signal_ma_cross(close.iloc[:1200], 10, 30, ma_type="EMA")
gap = float(np.nanmax(np.abs(ema_full.iloc[:1200].to_numpy() - ema_trunc.to_numpy())))
check("causality: EMA cross unchanged by future data", gap < 1e-10, f"gap={gap:.2e}")

vol_series = pd.Series(1000.0, index=idx)
vol_series.iloc[::2] = 100.0  # every other bar has low volume
pos_all = pd.Series(1.0, index=idx)
filt_v = bt.filter_volume(pos_all, vol_series, 20, 1.0)
check("volume filter zeroes low-volume bars",
      float(filt_v.iloc[100]) == 0.0 and float(filt_v.iloc[101]) == 1.0)

trend_pos = bt.filter_trend(pd.Series(1.0, index=down.index), down, 50)
check("trend filter kills longs below the MA", float(trend_pos.iloc[100:].abs().sum()) == 0.0)
short_pos = bt.filter_trend(pd.Series(-1.0, index=down.index), down, 50)
check("trend filter keeps shorts below the MA", float(short_pos.iloc[100:].min()) == -1.0)

r_1x = bt.run_backtest(close, sig, cost_bps=10.0, leverage=1.0)
r_2x = bt.run_backtest(close, sig, cost_bps=10.0, leverage=2.0)
gap2 = float(np.max(np.abs(r_2x.returns - 2.0 * r_1x.returns)))
check("lot ×2 doubles per-bar P&L (incl. costs)", gap2 < 1e-12, f"gap={gap2:.2e}")
check("equity in dollars scales with capital",
      abs(r_1x.equity_usd.iloc[-1] / 10_000.0 - r_1x.equity.iloc[-1]) < 1e-12)

m_daily = bt.perf_metrics(const, periods_per_year=252)
m_hourly = bt.perf_metrics(const, periods_per_year=252 * 6.5)
check("annualization scales with timeframe",
      abs(m_hourly["sharpe"] / m_daily["sharpe"] - np.sqrt(6.5)) < 1e-9,
      f"ratio={m_hourly['sharpe'] / m_daily['sharpe']:.4f} vs √6.5={np.sqrt(6.5):.4f}")

# --- Monthly table & rolling sharpe -------------------------------------------------
tbl = bt.monthly_return_table(bh.returns)
check("monthly table has 12 month columns", list(tbl.columns) == list(range(1, 13)))
rs = bt.rolling_sharpe(bh.returns, 126)
check("rolling sharpe defined after window", bool(np.isfinite(rs.iloc[200])))

# --- Walk-forward optimizer ----------------------------------------------------------
from quantsphere import optimize as qo  # noqa: E402

trend_px = pd.Series(
    100.0 * np.exp(np.cumsum(np.random.default_rng(7).normal(0.0006, 0.01, 1500))),
    index=idx)
res_o = qo.grid_search(trend_px, "MA crossover", cost_bps=2.0, rf=0.0, ppy=252,
                       long_short=False, train_frac=0.7)
check("optimizer returns ranked results", len(res_o) > 50
      and res_o["sharpe_test"].iloc[0] == res_o["sharpe_test"].max(),
      f"{len(res_o)} configs, best test Sharpe {res_o['sharpe_test'].iloc[0]:.2f}")
check("optimizer keeps both scores",
      {"sharpe_train", "sharpe_test", "cagr_test", "mdd_test"} <= set(res_o.columns))
check("split date recorded inside sample",
      idx[0] < pd.Timestamp(res_o.attrs["split_date"]) < idx[-1])

piv = qo.surface_pivot(res_o, "MA crossover")
check("surface pivot is 2-D over (fast, slow)",
      piv is not None and piv.shape[0] > 3 and piv.shape[1] > 3,
      f"shape={None if piv is None else piv.shape}")
check("1-D strategy has no pivot",
      qo.surface_pivot(res_o.assign(resp=0.0), "Kalman trend (drift sign)") is None)

res_k = qo.grid_search(trend_px, "Kalman trend (drift sign)", cost_bps=2.0, rf=0.0,
                       ppy=252, long_short=True, train_frac=0.7)
check("Kalman sweep runs on native engine", len(res_k) >= 20,
      f"{len(res_k)} configs")

# The optimizer's test metrics must equal an independent backtest of the same
# params scored on the test segment only (proves the split leaks nothing).
b0 = res_o.iloc[0]
sig_b = qo.build_signal(trend_px, "MA crossover",
                        {"fast": int(b0["fast"]), "slow": int(b0["slow"]),
                         "ma_type": b0["ma_type"]}, False, 252)
full = bt.run_backtest(trend_px, sig_b, cost_bps=2.0)
cut = int(len(full.returns) * 0.7)
m_ind = bt.perf_metrics(full.returns.iloc[cut:], 0.0, 252)
check("optimizer test score == independent replay",
      abs(m_ind["sharpe"] - b0["sharpe_test"]) < 1e-9,
      f"{m_ind['sharpe']:.4f} vs {b0['sharpe_test']:.4f}")

print()
if FAILURES:
    print(f"❌ {len(FAILURES)} failure(s): {FAILURES}")
    sys.exit(1)
print("✅ ALL QUANT CHECKS PASSED")
