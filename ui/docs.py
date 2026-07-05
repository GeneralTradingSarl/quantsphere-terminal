"""📚 Full mathematical documentation of every model, formula and numerical
scheme used in QuantSphere Terminal. Rendered with st.latex — this is the
methodology the platform can be audited against."""

from __future__ import annotations

import streamlit as st


def _h(txt: str) -> None:
    st.markdown(f"##### {txt}")


def render_docs() -> None:
    st.markdown("## 📚 Models & Methodology")
    st.markdown(
        "Complete specification of every quantity displayed in the terminal: "
        "model equations, estimators, numerical schemes and their convergence "
        "properties. Notation: $S_t$ spot price, $r$ risk-free rate, $\\sigma$ "
        "volatility, $W_t$ standard Brownian motion, $\\Delta = 252$ trading "
        "days per year."
    )

    # ------------------------------------------------------------------ 1
    with st.expander("1 · Market data & empirical statistics", expanded=False):
        _h("Log-returns and annualization")
        st.latex(r"x_t = \ln\frac{S_t}{S_{t-1}}, \qquad \hat\mu = 252\,\bar{x}, \qquad \hat\sigma = \sqrt{252}\; s_x")
        st.markdown("where $\\bar x$ and $s_x$ are the sample mean and standard deviation "
                    "(ddof = 1) of daily log-returns.")
        _h("Higher moments")
        st.latex(r"\text{skew} = \frac{\frac{1}{n}\sum (x_t-\bar x)^3}{s_x^3},\qquad \text{excess kurtosis} = \frac{\frac{1}{n}\sum (x_t-\bar x)^4}{s_x^4} - 3")
        st.markdown("A Gaussian has both equal to 0. Positive excess kurtosis (fat tails) "
                    "is the empirical motivation for the jump models of Stage 4.")
        _h("Historical Value-at-Risk (1-day, 95%)")
        st.latex(r"\text{VaR}_{95\%} = -\,Q_{0.05}(x_1,\dots,x_n)")
        st.markdown("the empirical 5% quantile of daily log-returns, reported as a "
                    "positive loss. Realized volatility uses a rolling 21-day window: "
                    "$\\sigma^{(21)}_t = \\sqrt{252}\\,\\mathrm{sd}(x_{t-20..t})$.")
        _h("Intraday session & VWAP")
        st.latex(r"\text{VWAP}_t = \frac{\sum_{i \le t} P_i V_i}{\sum_{i \le t} V_i} \qquad \text{(reset at each session open)}")
        st.markdown(
            "Intraday charts remove weekends and non-trading hours from the time "
            "axis (range breaks), so overnight gaps do not draw as false moves. "
            "Prices are dividend/split adjusted; zero-volume bars are excluded from "
            "the VWAP. Data-source lookback caps: 1m → 7 days, 5m/15m/30m → 60 "
            "days, 1h → 730 days; quotes can lag a few minutes and show the last "
            "completed session when the market is closed.")

    # ------------------------------------------------------------------ 2
    with st.expander("2 · Kalman filter & RTS smoother"):
        _h("State-space form (scalar observation)")
        st.latex(r"\mathbf{x}_t = A\,\mathbf{x}_{t-1} + \mathbf{w}_t,\quad \mathbf{w}_t \sim \mathcal N(0, Q)")
        st.latex(r"y_t = C\,\mathbf{x}_t + v_t,\quad v_t \sim \mathcal N(0, R)")
        st.markdown(
            "**Local level** (dim 1): $A=1,\\ C=1$ — the price is a random walk "
            "observed in noise. **Local linear trend** (dim 2): state "
            "$[\\text{level},\\ \\text{drift}]$ with "
            "$A=\\begin{pmatrix}1&1\\\\0&1\\end{pmatrix}$, $C=(1\\ 0)$ — the drift "
            "state is the latent alpha the strategy tab trades on.")
        _h("Predict / update recursions")
        st.latex(r"\hat{\mathbf{x}}_{t|t-1} = A\hat{\mathbf{x}}_{t-1|t-1}, \qquad P_{t|t-1} = A P_{t-1|t-1} A^\top + Q")
        st.latex(r"S_t = C P_{t|t-1} C^\top + R, \qquad K_t = P_{t|t-1} C^\top S_t^{-1}")
        st.latex(r"\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + K_t\,\nu_t, \qquad P_{t|t} = (I - K_t C)P_{t|t-1}")
        st.markdown("with innovation $\\nu_t = y_t - C\\hat{\\mathbf x}_{t|t-1}$. The "
                    "log-likelihood accumulates "
                    "$-\\tfrac12(\\ln 2\\pi + \\ln S_t + \\nu_t^2/S_t)$.")
        _h("Rauch–Tung–Striebel smoother (two-sided)")
        st.latex(r"G_t = P_{t|t} A^\top P_{t+1|t}^{-1}")
        st.latex(r"\hat{\mathbf{x}}_{t|n} = \hat{\mathbf{x}}_{t|t} + G_t(\hat{\mathbf{x}}_{t+1|n} - \hat{\mathbf{x}}_{t+1|t})")
        st.markdown(
            "⚠️ The smoother uses future data, so it is shown for analysis but is "
            "**never** used in backtest signals (only the causal filter is).")
        _h("Noise calibration & diagnostics")
        st.latex(r"\hat R = \tfrac12\,\mathrm{Var}(\Delta y_t), \qquad Q = \hat R \cdot 10^{\text{responsiveness}}")
        st.markdown("If the model is correct, standardized innovations "
                    "$\\nu_t/\\sqrt{S_t}$ are i.i.d. $\\mathcal N(0,1)$ — the "
                    "whiteness histogram checks exactly this.")
        _h("h-step-ahead prediction (the dashed forecast on the chart)")
        st.latex(r"\hat{\mathbf x}_{t+h|t} = A^h\, \hat{\mathbf x}_{t|t}, \qquad P_{t+h|t} = A\,P_{t+h-1|t}\,A^\top + Q")
        st.markdown(
            "For the local trend model the point forecast is "
            "$\\text{level}_t + h\\cdot\\text{drift}_t$ (for the local level, it is "
            "flat at the current level — a random walk has no predictable "
            "direction). The ±2σ band widens as the state covariance compounds "
            "through $Q$: the further out, the humbler the forecast. This is a "
            "*conditional expectation*, not a trading signal by itself.")

    # ------------------------------------------------------------------ 3
    with st.expander("3 · Black-Scholes & implied volatility"):
        _h("Black-Scholes closed form")
        st.latex(r"C = S\,\Phi(d_1) - K e^{-rT}\Phi(d_2), \qquad P = K e^{-rT}\Phi(-d_2) - S\,\Phi(-d_1)")
        st.latex(r"d_{1,2} = \frac{\ln(S/K) + (r \pm \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}")
        _h("Implied volatility")
        st.markdown("Given a market mid quote $\\hat C$ (mid $=(\\text{bid}+\\text{ask})/2$ "
                    "when two-sided), $\\sigma_{imp}$ solves "
                    "$BS(\\sigma) = \\hat C$ by Newton-Raphson with analytic vega:")
        st.latex(r"\sigma_{k+1} = \sigma_k - \frac{BS(\sigma_k) - \hat C}{\mathcal V(\sigma_k)}, \qquad \mathcal V = S\,\varphi(d_1)\sqrt{T}")
        st.markdown(
            "started at the Brenner-Subrahmanyam value "
            "$\\sigma_0 \\approx \\sqrt{2\\pi/T}\\,\\hat C/S$, with a guaranteed "
            "bisection fallback on $[10^{-6}, 5]$ (price is monotone in $\\sigma$). "
            "Quotes outside the no-arbitrage band return NaN and are discarded. "
            "The surface is built from **OTM** quotes only (calls above spot, puts "
            "below) and interpolated linearly in $(K, T)$; the *flat-vol RMSE* "
            "metric is $\\sqrt{\\mathbb E[(\\sigma_{imp} - \\sigma_{ATM})^2]}$ — "
            "the smile Black-Scholes cannot reproduce.")

    # ------------------------------------------------------------------ 4
    with st.expander("4 · Price dynamics: GBM, Heston, Merton, Variance Gamma"):
        st.markdown("All simulations are **risk-neutral** (drift $r$) with exact "
                    "martingale corrections; discretization uses $n = 252\\,T$ steps.")
        _h("Geometric Brownian motion (exact scheme)")
        st.latex(r"dS_t = r S_t\,dt + \sigma S_t\,dW_t 	\;\Rightarrow\; \ln S_{t+\Delta t} = \ln S_t + (r - \tfrac12\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z")
        _h("Heston stochastic volatility (full-truncation Euler)")
        st.latex(r"dS_t = r S_t dt + \sqrt{v_t}\,S_t\,dW^S_t, \qquad dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}\,dW^v_t")
        st.latex(r"d\langle W^S, W^v\rangle_t = \rho\,dt, \qquad v^+ = \max(v, 0)\ \text{in drift and diffusion}")
        st.markdown("Full truncation (Lord–Koekkoek–van Dijk 2010) is the standard "
                    "bias-minimizing fix when the Feller condition "
                    "$2\\kappa\\theta > \\xi^2$ fails.")
        _h("Merton jump-diffusion (compensated)")
        st.latex(r"\frac{dS_t}{S_{t^-}} = (r - \lambda k)\,dt + \sigma\,dW_t + (e^{J}-1)\,dN_t")
        st.latex(r"N_t \sim \text{Poisson}(\lambda t),\quad J \sim \mathcal N(\mu_J, \sigma_J^2),\quad k = e^{\mu_J + \sigma_J^2/2} - 1")
        _h("Variance Gamma (gamma time change)")
        st.latex(r"X_t = \theta\,G_t + \sigma\,W_{G_t}, \qquad G_{t+\Delta t}-G_t \sim \Gamma(\Delta t/\nu,\ \nu)")
        st.latex(r"S_t = S_0\,\exp\big((r+\omega)t + X_t\big), \qquad \omega = \tfrac{1}{\nu}\ln\!\big(1 - \theta\nu - \tfrac12\sigma^2\nu\big)")
        st.markdown("$\\omega$ is the exponential compensator keeping "
                    "$e^{-rt}S_t$ a martingale; the parameter constraint "
                    "$1-\\theta\\nu-\\tfrac12\\sigma^2\\nu > 0$ is enforced.")
        _h("Monte Carlo estimator & error")
        st.latex(r"\hat V = e^{-rT}\,\frac1N \sum_{i=1}^N \text{payoff}_i, \qquad \text{SE} = e^{-rT}\sqrt{\frac{\widehat{\mathrm{Var}}(\text{payoff})}{N}}")
        st.markdown("Barrier options monitor the discrete grid (Broadie-Glasserman "
                    "continuity correction not applied — discrete monitoring is "
                    "stated explicitly). In/out parity $V_{KO} + V_{KI} = V_{vanilla}$ "
                    "holds path-wise and is verified in the test suite. RNG: 64 fixed "
                    "work units seeded via splitmix64 → results are deterministic and "
                    "independent of thread count.")

    # ------------------------------------------------------------------ 5
    with st.expander("5 · PDE engine: Crank-Nicolson, Thomas, PSOR"):
        _h("Black-Scholes PDE")
        st.latex(r"\frac{\partial V}{\partial t} + \tfrac12\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0")
        _h("Crank-Nicolson discretization")
        st.markdown("Uniform grid $S_i = i\\,\\Delta S$, $t_j = j\\,\\Delta t$; "
                    "averaging the explicit and implicit operators gives the "
                    "unconditionally stable, $O(\\Delta t^2 + \\Delta S^2)$ scheme")
        st.latex(r"(I - \tfrac12 M)\,V^{j} = (I + \tfrac12 M)\,V^{j+1}")
        st.latex(r"a_i = \tfrac{\Delta t}{4}(\sigma^2 i^2 - r i),\quad b_i = -\tfrac{\Delta t}{2}(\sigma^2 i^2 + r),\quad c_i = \tfrac{\Delta t}{4}(\sigma^2 i^2 + r i)")
        st.markdown("The domain is truncated at "
                    "$S_{max} = \\max(3, e^{3\\sigma\\sqrt T})\\max(S_0,K)$ so the "
                    "3-standard-deviation diffusion cone stays inside the grid. "
                    "Dirichlet boundaries: e.g. call: $V(0,t)=0$, "
                    "$V(S_{max},t) = S_{max} - Ke^{-r\\tau}$.")
        _h("Solvers")
        st.markdown(
            "**European**: the tridiagonal system is solved by the Thomas algorithm "
            "in $O(N_s)$ per step. **American**: early exercise turns the PDE into "
            "a linear complementarity problem")
        st.latex(r"\min\big(\mathcal L V,\; V - \psi\big) = 0, \qquad \psi = \text{payoff}")
        st.markdown("solved by **PSOR** (projected successive over-relaxation, "
                    "$\\omega = 1.2$, tol $10^{-9}$) in the C++ engine, and by "
                    "operator-splitting projection $V \\leftarrow \\max(V, \\psi)$ in "
                    "the NumPy fallback (both agree to ~$10^{-3}$ relative). Greeks "
                    "are central finite differences on the $t=0$ layer: "
                    "$\\Delta = \\frac{V_{i+1}-V_{i-1}}{2\\Delta S}$, "
                    "$\\Gamma = \\frac{V_{i+1}-2V_i+V_{i-1}}{\\Delta S^2}$, "
                    "$\\Theta = \\frac{V(S_0,\\Delta t)-V(S_0,0)}{\\Delta t}$.")

    # ------------------------------------------------------------------ 6
    with st.expander("6 · Backtesting methodology & performance metrics"):
        _h("Execution discipline (no look-ahead)")
        st.latex(r"R^{strat}_t = w_{t-1}\,R_t \; - \; \text{cost}_t, \qquad \text{cost}_t = |w_{t-1} - w_{t-2}| \cdot \frac{\text{bps}}{10^4}")
        st.markdown(
            "A position $w$ decided at the close of bar $t-1$ (using only data up "
            "to $t-1$) earns the return of bar $t$ — a one-bar execution lag. "
            "Transaction costs are charged on turnover. Kalman signals use the "
            "**filtered** (causal) state only, never the smoother.")
        _h("Strategy signals")
        st.markdown(
            "- **Kalman trend**: $w_t = \\mathrm{sign}(\\hat\\beta_{t|t})$ on the "
            "filtered drift of the local-trend model on $\\ln S$.\n"
            "- **MA crossover**: $w_t = \\mathbb 1[\\text{MA}_f > \\text{MA}_s]$ "
            "(or $\\pm 1$ long/short), with SMA or EMA "
            "($\\text{EMA}_t = \\lambda S_t + (1-\\lambda)\\text{EMA}_{t-1}$, "
            "$\\lambda = 2/(n+1)$).\n"
            "- **Mean reversion**: $z_t = \\frac{S_t - \\text{SMA}_n}{\\text{sd}_n}$; "
            "$w_t = -\\mathrm{sign}(z_t)\\,\\mathbb 1[|z_t| > z_{entry}]$.\n"
            "- **Vol targeting**: $w_t = \\min\\!\\big(\\frac{\\sigma^{target}}"
            "{\\sigma^{(n)}_t}, L_{max}\\big)$ — constant-risk exposure.")
        _h("Composable position filters (both causal)")
        st.markdown(
            "- **Trend regime**: longs allowed only when $S_t > \\text{MA}_n(S)$, "
            "shorts only below — the classic regime overlay.\n"
            "- **Volume confirmation**: position held only when "
            "$V_t \\ge m \\cdot \\overline{V}_{20}$ — trades are confirmed by "
            "participation.\n"
            "- **Lot / leverage** $L$: every exposure and every cost is scaled by "
            "$L$; equity compounds on $L\\,w_{t-1} R_t - L\\,|\\Delta w|\\,"
            "\\text{bps}\\cdot 10^{-4}$.")
        _h("Timeframe-aware annualization")
        st.latex(r"N_{yr} \in \{252,\ 1638,\ 3276,\ 6552,\ 19656\} \ \text{bars for } \{1d, 1h, 30m, 15m, 5m\}")
        st.markdown("(US cash session ≈ 6.5 h). All Sharpe/vol/CAGR figures scale "
                    "with $\\sqrt{N_{yr}}$ or $N_{yr}$ accordingly; intraday history "
                    "is capped by the data source (1h → 2 years, others → 60 days).")
        _h("Metrics")
        st.latex(r"\text{CAGR} = E_n^{252/n} - 1, \qquad \text{Sharpe} = \frac{252\,\bar R - r_f}{\sqrt{252}\,s_R}")
        st.latex(r"\text{Sortino} = \frac{252\,\bar R - r_f}{\sqrt{252}\; \mathrm{sd}(R \mid R<0)}, \qquad \text{MDD} = \min_t \Big(\frac{E_t}{\max_{s\le t} E_s} - 1\Big)")
        st.latex(r"\text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|}, \qquad \text{hit rate} = \Pr(R_t > 0 \mid w_{t-1} \ne 0)")
        st.markdown("⚠️ Standard caveats apply and are part of the methodology: "
                    "single-asset in-sample results, no borrowing costs or "
                    "short-availability constraints, daily close fills. The tool is "
                    "a research instrument, not investment advice.")

    # ------------------------------------------------------------------ 7
    with st.expander("7 · Volatility forecasting: GARCH(1,1) & EWMA"):
        _h("GARCH(1,1) — Bollerslev (1986)")
        st.latex(r"r_t = \sigma_t \varepsilon_t,\quad \varepsilon_t \sim \mathcal N(0,1), \qquad \sigma^2_t = \omega + \alpha\, r^2_{t-1} + \beta\, \sigma^2_{t-1}")
        _h("Maximum likelihood estimation")
        st.latex(r"\ln \mathcal L = -\tfrac12 \sum_{t}\Big(\ln 2\pi + \ln \sigma^2_t + \frac{r_t^2}{\sigma^2_t}\Big)")
        st.markdown("maximized by Nelder-Mead from three starting points, under "
                    "$\\omega > 0,\\ \\alpha,\\beta \\ge 0,\\ \\alpha + \\beta < 1$ "
                    "(covariance stationarity). Returns are demeaned and scaled "
                    "×100 for conditioning.")
        _h("Forecast term structure & half-life")
        st.latex(r"\bar\sigma^2 = \frac{\omega}{1 - \alpha - \beta}, \qquad \mathbb E[\sigma^2_{t+h}] = \bar\sigma^2 + (\alpha+\beta)^{h-1}\big(\sigma^2_{t+1} - \bar\sigma^2\big)")
        st.latex(r"\text{half-life} = \frac{\ln 0.5}{\ln(\alpha+\beta)} \ \text{days}")
        _h("RiskMetrics EWMA (benchmark)")
        st.latex(r"\sigma^2_t = \lambda\,\sigma^2_{t-1} + (1-\lambda)\,r^2_{t-1}, \qquad \lambda = 0.94")

    # ------------------------------------------------------------------ 8
    with st.expander("8 · Price forecasting: cones, bootstrap, density"):
        _h("Analytic GBM cone (real-world drift)")
        st.latex(r"S_t^{(q)} = S_0 \exp\Big((\mu - \tfrac12\sigma^2)t + \sigma\sqrt{t}\; z_q\Big), \qquad z_q = \Phi^{-1}(q)")
        st.markdown("plotted for $q \\in \\{5, 25, 50, 75, 95\\}\\%$. Drift choices: "
                    "empirical $\\hat\\mu$, risk-free $r$, or zero (a pure "
                    "uncertainty cone). Volatility: historical $\\hat\\sigma$ or the "
                    "GARCH forecast term structure.")
        _h("Circular block bootstrap (model-free)")
        st.markdown("Future log-returns are resampled in blocks of $b=5$ days from "
                    "the empirical distribution (circular indexing), preserving fat "
                    "tails and short-range autocorrelation without assuming "
                    "normality. 10,000 paths; the cone is the per-day empirical "
                    "quantile.")
        _h("Forward density surface")
        st.latex(r"p(S, t) = \frac{1}{S\,\sigma\sqrt{2\pi t}}\; \exp\!\left(-\frac{\big(\ln(S/S_0) - (\mu - \tfrac12\sigma^2)t\big)^2}{2\sigma^2 t}\right)")
        _h("Horizon probabilities & tail risk")
        st.latex(r"\Pr(S_T > x) = 1 - \Phi\!\left(\frac{\ln(x/S_0) - (\mu-\tfrac12\sigma^2)T}{\sigma\sqrt T}\right)")
        st.latex(r"\text{VaR}_\alpha = -Q_\alpha\big(S_T/S_0 - 1\big), \qquad \text{ES}_\alpha = -\,\mathbb E\big[S_T/S_0 - 1 \,\big|\, \cdot \le Q_\alpha\big]")

    # ------------------------------------------------------------------ 9
    with st.expander("9 · Markowitz portfolio optimization"):
        _h("Inputs")
        st.latex(r"\boldsymbol\mu = 252\,\overline{\mathbf{x}}, \qquad \Sigma = 252\,\widehat{\mathrm{Cov}}(\mathbf{x}) + \epsilon I")
        st.markdown("with ridge $\\epsilon$ increased until $\\Sigma \\succ 0$ "
                    "(Cholesky test) — a near-singular basket cannot break the solver.")
        _h("Programs (long-only, fully invested)")
        st.latex(r"\text{Max Sharpe:}\quad \max_{w}\; \frac{w^\top\boldsymbol\mu - r_f}{\sqrt{w^\top \Sigma w}} \quad \text{s.t.}\; \mathbf 1^\top w = 1,\; w \ge 0")
        st.latex(r"\text{Min variance:}\quad \min_w\; w^\top \Sigma w, \qquad \text{Frontier:}\quad \min_w w^\top\Sigma w \;\text{ s.t. } w^\top\boldsymbol\mu = \mu^{target}")
        st.markdown("solved by SLSQP; the frontier is 40 constrained quadratic "
                    "programs. The sampled cloud is Dirichlet(1) — uniform on the "
                    "simplex. The capital market line through the tangent portfolio: "
                    "$\\mu = r_f + \\text{Sharpe}^* \\cdot \\sigma$.")

    # ------------------------------------------------------------------ 10
    with st.expander("10 · Assumptions, limitations & model risk"):
        st.markdown(
            "Every number in this terminal is conditional on assumptions. The ones "
            "that matter:\n\n"
            "- **Data**: Yahoo Finance quotes — adjusted for splits/dividends, but "
            "subject to vendor errors, survivorship of current tickers, and the "
            "intraday lookback caps of §1. Option quotes are mid prices; illiquid "
            "strikes (no bid, no volume/OI) are filtered out but stale quotes can "
            "survive.\n"
            "- **Gaussian innovations**: the Kalman filter and GARCH likelihood "
            "assume normal shocks. Real returns have fat tails (see the kurtosis "
            "metric) — the jump models and the block bootstrap exist precisely to "
            "quantify what normality misses.\n"
            "- **Risk-neutral vs real-world**: pricing (Stage 4) simulates under "
            "the risk-neutral measure (drift $r$); forecasting (Stage 7) uses a "
            "real-world drift you choose. The two answer different questions — "
            "never read a pricing path fan as a prediction.\n"
            "- **Backtests are in-sample research**: signals and their parameters "
            "are chosen while looking at the whole chart. The engine guarantees no "
            "*mechanical* look-ahead (1-bar lag, causal filters — verified by "
            "test), but it cannot protect against parameter overfitting. Costs are "
            "a flat bps rate: no market impact, borrow fees, or overnight "
            "financing.\n"
            "- **Discrete barrier monitoring**: knock-in/out events are checked on "
            "the simulation grid, which slightly underprices continuous barriers "
            "(Broadie-Glasserman correction not applied, stated by design).\n"
            "- **Optimization inputs**: Markowitz weights are extremely sensitive "
            "to $\\hat\\mu$; five years of data estimates an annual mean with a "
            "standard error of roughly $\\sigma/\\sqrt{5}$ — often as large as the "
            "estimate itself. Treat the frontier as a map, not a prescription.\n\n"
            "**Nothing here is investment advice.** The platform is a research and "
            "education instrument; its value is that every formula above is stated, "
            "implemented twice (C++ and NumPy), and cross-verified by an automated "
            "suite.")

    # ------------------------------------------------------------------ refs
    with st.expander("References"):
        st.markdown(
            "- Black, F. & Scholes, M. (1973), *The Pricing of Options and "
            "Corporate Liabilities*, JPE 81(3).\n"
            "- Merton, R. (1976), *Option pricing when underlying stock returns "
            "are discontinuous*, JFE 3.\n"
            "- Heston, S. (1993), *A Closed-Form Solution for Options with "
            "Stochastic Volatility*, RFS 6(2).\n"
            "- Madan, D., Carr, P. & Chang, E. (1998), *The Variance Gamma Process "
            "and Option Pricing*, European Finance Review 2.\n"
            "- Lord, R., Koekkoek, R. & van Dijk, D. (2010), *A comparison of "
            "biased simulation schemes for stochastic volatility models*, "
            "Quantitative Finance 10(2).\n"
            "- Bollerslev, T. (1986), *Generalized Autoregressive Conditional "
            "Heteroskedasticity*, Journal of Econometrics 31.\n"
            "- Harvey, A. (1989), *Forecasting, Structural Time Series Models and "
            "the Kalman Filter*, CUP.\n"
            "- Markowitz, H. (1952), *Portfolio Selection*, Journal of Finance 7(1).\n"
            "- Wilmott, Howison & Dewynne (1995), *The Mathematics of Financial "
            "Derivatives*, CUP (Crank-Nicolson, PSOR).\n"
            "- Politis, D. & Romano, J. (1992), circular block bootstrap.\n"
            "- Cantarutti, N., *Financial-Models-Numerical-Methods* — the notebook "
            "collection this platform industrializes.")

    st.markdown(
        "---\n**QuantSphere Terminal** — © 2026 Ismael LADJOHOUNLOU · Licensed "
        "under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html): free to "
        "use and study; any modified version that is distributed **or served "
        "over a network** must publish its complete source under the same "
        "license. For commercial licensing, contact the author.")
