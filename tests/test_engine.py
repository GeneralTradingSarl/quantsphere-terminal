"""QuantSphere engine verification suite — network-free.

Checks the native C++ engine and the NumPy fallback against closed forms and
against each other. Run:  python tests/test_engine.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantsphere import ENGINE_NATIVE, core  # noqa: E402
from quantsphere import _fallback as fb  # noqa: E402
from quantsphere import portfolio as qsport  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


ENGINES = [("native" if ENGINE_NATIVE else "fallback-as-core", core), ("fallback", fb)]

# --- Black-Scholes closed form ------------------------------------------------
for label, eng in ENGINES:
    c = eng.bs_price(100, 100, 1.0, 0.05, 0.2, True)
    p = eng.bs_price(100, 100, 1.0, 0.05, 0.2, False)
    check(f"BS known value [{label}]", abs(c - 10.450584) < 1e-4, f"C={c:.6f}")
    parity = c - p - (100 - 100 * np.exp(-0.05))
    check(f"put-call parity [{label}]", abs(parity) < 1e-9, f"gap={parity:.2e}")

# --- Implied vol round-trip ----------------------------------------------------
for label, eng in ENGINES:
    sig_true = np.array([0.15, 0.35, 0.8])
    Ks = np.array([90.0, 100.0, 120.0])
    Ts = np.array([0.25, 1.0, 2.0])
    calls = np.array([True, False, True])
    prices = np.array([eng.bs_price(100, k, t, 0.03, s, c)
                       for k, t, s, c in zip(Ks, Ts, sig_true, calls)])
    iv = np.asarray(eng.implied_vol(prices, 100.0, Ks, Ts, 0.03, calls))
    check(f"IV round-trip [{label}]", np.allclose(iv, sig_true, atol=1e-6),
          f"max err={np.abs(iv - sig_true).max():.2e}")
    bad = np.asarray(eng.implied_vol(np.array([200.0]), 100.0, np.array([100.0]),
                                     np.array([1.0]), 0.03, np.array([True])))
    check(f"IV rejects arbitrage [{label}]", bool(np.isnan(bad[0])))

# --- Monte Carlo vs closed form -----------------------------------------------
BS_REF = core.bs_price(100, 105, 1.0, 0.03, 0.25, True)
for label, eng in ENGINES:
    r = eng.mc_price("gbm", 105, True, n_paths=200_000, steps=252, seed=11,
                     S0=100, r=0.03, T=1.0, sigma=0.25)
    err = abs(r["price"] - BS_REF)
    check(f"MC GBM vs BS [{label}]", err < 3.5 * r["std_error"] + 1e-9,
          f"{r['price']:.4f} vs {BS_REF:.4f}, SE={r['std_error']:.4f}")

    m = eng.mc_price("merton", 105, True, n_paths=200_000, steps=252, seed=11,
                     S0=100, r=0.03, T=1.0, sigma=0.25, lam=0.0)
    check(f"Merton(λ=0) == GBM [{label}]", abs(m["price"] - BS_REF) < 3.5 * m["std_error"],
          f"{m['price']:.4f}")

    h = eng.mc_price("heston", 105, True, n_paths=200_000, steps=252, seed=11,
                     S0=100, r=0.03, T=1.0, v0=0.0625, kappa=5.0, theta=0.0625,
                     xi=1e-4, rho=0.0)
    check(f"Heston(ξ→0) ≈ BS [{label}]", abs(h["price"] - BS_REF) < 4.0 * h["std_error"] + 0.02,
          f"{h['price']:.4f}")

    v = eng.mc_price("vg", 105, True, n_paths=200_000, steps=252, seed=11,
                     S0=100, r=0.03, T=1.0, sigma=0.25, vg_theta=0.0, vg_nu=0.005)
    check(f"VG(ν→0) ≈ BS [{label}]", abs(v["price"] - BS_REF) < 4.0 * v["std_error"] + 0.05,
          f"{v['price']:.4f}")

# --- Barrier in/out parity (same seed => exact path-wise decomposition) --------
for label, eng in ENGINES:
    kw = dict(n_paths=50_000, steps=252, seed=5, S0=100, r=0.03, T=1.0, sigma=0.25)
    van = eng.mc_price("gbm", 100, True, barrier_type="none", **kw)
    uo = eng.mc_price("gbm", 100, True, barrier_type="up-and-out", barrier=130.0, **kw)
    ui = eng.mc_price("gbm", 100, True, barrier_type="up-and-in", barrier=130.0, **kw)
    gap = abs(uo["price"] + ui["price"] - van["price"])
    check(f"barrier in-out parity [{label}]", gap < 1e-9, f"gap={gap:.2e}")
    check(f"knock-out ≤ vanilla [{label}]", uo["price"] <= van["price"] + 1e-12)

# --- PDE vs closed form ---------------------------------------------------------
for label, eng in ENGINES:
    for is_call in (True, False):
        pd_res = eng.pde_price(100, 105, 0.03, 0.25, 1.0, is_call=is_call,
                               american=False, Ns=400, Nt=400)
        ref = core.bs_price(100, 105, 1.0, 0.03, 0.25, is_call)
        rel = abs(pd_res["price"] - ref) / ref
        check(f"PDE European {'call' if is_call else 'put'} vs BS [{label}]",
              rel < 2e-3, f"{pd_res['price']:.4f} vs {ref:.4f} ({rel:.2e})")

    am = eng.pde_price(100, 105, 0.03, 0.25, 1.0, is_call=False, american=True,
                       Ns=400, Nt=400)
    eu = eng.pde_price(100, 105, 0.03, 0.25, 1.0, is_call=False, american=False,
                       Ns=400, Nt=400)
    check(f"American put ≥ European put [{label}]", am["price"] >= eu["price"] - 1e-9,
          f"{am['price']:.4f} ≥ {eu['price']:.4f}")
    check(f"American put ≥ intrinsic [{label}]", am["price"] >= 5.0 - 1e-9)

if ENGINE_NATIVE:
    a = core.pde_price(100, 105, 0.03, 0.25, 1.0, is_call=True, american=False,
                       Ns=300, Nt=300)
    b = fb.pde_price(100, 105, 0.03, 0.25, 1.0, is_call=True, american=False,
                     Ns=300, Nt=300)
    check("PDE native == fallback (European)", abs(a["price"] - b["price"]) < 1e-8,
          f"gap={abs(a['price'] - b['price']):.2e}")
    am_a = core.pde_price(100, 105, 0.03, 0.25, 1.0, is_call=False, american=True,
                          Ns=300, Nt=300)
    am_b = fb.pde_price(100, 105, 0.03, 0.25, 1.0, is_call=False, american=True,
                        Ns=300, Nt=300)
    check("PDE native ≈ fallback (American, PSOR vs projection)",
          abs(am_a["price"] - am_b["price"]) / am_a["price"] < 1e-2,
          f"{am_a['price']:.4f} vs {am_b['price']:.4f}")

# --- Kalman filter ---------------------------------------------------------------
rng = np.random.default_rng(3)
n = 800
truth = np.cumsum(rng.normal(0, 0.05, n)) + 100.0
noisy = truth + rng.normal(0, 0.8, n)
for label, eng in ENGINES:
    kf = eng.kalman(noisy, model="local_level", q=0.05**2, r=0.8**2)
    filt = np.asarray(kf["x_filt"])[:, 0]
    smooth = np.asarray(kf["x_smooth"])[:, 0]
    mse_raw = float(np.mean((noisy - truth) ** 2))
    mse_f = float(np.mean((filt[50:] - truth[50:]) ** 2))
    mse_s = float(np.mean((smooth[50:] - truth[50:]) ** 2))
    check(f"Kalman beats raw noise [{label}]", mse_f < 0.35 * mse_raw,
          f"raw={mse_raw:.3f} filt={mse_f:.3f}")
    check(f"RTS smoother beats filter [{label}]", mse_s <= mse_f + 1e-12,
          f"smooth={mse_s:.3f}")
    z = np.asarray(kf["innovations"])[10:] / np.sqrt(np.asarray(kf["innov_var"])[10:])
    check(f"innovations calibrated [{label}]", 0.8 < float(np.std(z)) < 1.2,
          f"std={np.std(z):.3f}")

if ENGINE_NATIVE:
    ka = core.kalman(noisy, model="local_trend", q=1e-3, r=0.64, q_drift=1e-6)
    kb = fb.kalman(noisy, model="local_trend", q=1e-3, r=0.64, q_drift=1e-6)
    gap = float(np.max(np.abs(np.asarray(ka["x_smooth"]) - np.asarray(kb["x_smooth"]))))
    check("Kalman native == fallback", gap < 1e-8, f"max gap={gap:.2e}")
    check("loglik native == fallback", abs(ka["loglik"] - kb["loglik"]) < 1e-6)

# --- Portfolio optimizer -----------------------------------------------------------
rng = np.random.default_rng(9)
n_assets = 6
A = rng.normal(size=(n_assets, n_assets))
cov = A @ A.T / 400.0 + np.eye(n_assets) * 1e-4
mu = rng.uniform(0.02, 0.15, n_assets)
w_msr = qsport.max_sharpe(mu, cov, rf=0.02)
w_mv = qsport.min_variance(mu, cov)
check("MSR weights valid", abs(w_msr.sum() - 1) < 1e-9 and (w_msr >= -1e-12).all())
check("MinVar weights valid", abs(w_mv.sum() - 1) < 1e-9 and (w_mv >= -1e-12).all())
cloud = qsport.random_portfolios(mu, cov, n=4000, rf=0.02)
ret, vol = qsport.portfolio_perf(w_msr, mu, cov)
check("MSR dominates sampled cloud", (ret - 0.02) / vol >= float(cloud["sharpes"].max()) - 1e-6,
      f"solver={(ret - 0.02) / vol:.4f} cloud max={cloud['sharpes'].max():.4f}")
_, vmv = qsport.portfolio_perf(w_mv, mu, cov)
check("MinVar dominates sampled cloud", vmv <= float(cloud["vols"].min()) + 1e-9)

# --- Verdict -------------------------------------------------------------------------
print()
if FAILURES:
    print(f"❌ {len(FAILURES)} failure(s): {FAILURES}")
    sys.exit(1)
print(f"✅ ALL CHECKS PASSED — engine: {'C++ native' if ENGINE_NATIVE else 'NumPy fallback'}")
