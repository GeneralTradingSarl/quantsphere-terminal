"""QuantSphere Terminal — institutional quantitative pipeline.

Five live stages: market data -> Kalman filtering -> volatility surface ->
derivative pricing (Monte Carlo + PDE) -> portfolio optimization.
Numerical core: C++20 engine (pybind11) with a NumPy fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from scipy.stats import norm

from quantsphere import ENGINE_LABEL, ENGINE_NATIVE, core
from quantsphere import backtest as qsbt
from quantsphere import data as qsdata
from quantsphere import models as qsmodels
from quantsphere import optimize as qsopt
from quantsphere import portfolio as qsport
from ui.docs import render_docs
from ui.theme import BLUE_GREEN_SCALE, C, PURPLE_SCALE, inject_css, render_header, style_fig

st.set_page_config(page_title="QuantSphere Terminal", page_icon="◈", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()

# ---------------------------------------------------------------------------
# Cached data boundary (pure functions live in quantsphere.data / .portfolio)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def cached_history(ticker: str, years: int) -> pd.DataFrame:
    return qsdata.fetch_history(ticker, years)


@st.cache_data(ttl=60, show_spinner=False)
def cached_intraday(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return qsdata.fetch_intraday(ticker, period, interval)


@st.cache_data(ttl=900, show_spinner=False)
def cached_chain(ticker: str) -> tuple[pd.DataFrame, float]:
    return qsdata.fetch_option_chain(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_bars(ticker: str, interval: str, years: int) -> pd.DataFrame:
    return qsdata.fetch_bars(ticker, interval, years)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_portfolio(tickers: tuple[str, ...], years: int, rf: float) -> dict:
    prices = qsport.fetch_basket(tickers, years)
    mu, cov = qsport.annual_stats(prices)
    cloud = qsport.random_portfolios(mu, cov, n=4000, rf=rf)
    w_msr = qsport.max_sharpe(mu, cov, rf)
    w_mv = qsport.min_variance(mu, cov)
    frontier = qsport.efficient_frontier(mu, cov, n_points=40)
    return {
        "tickers": tickers, "mu": mu, "cov": cov, "cloud": cloud,
        "w_msr": w_msr, "w_mv": w_mv, "frontier": frontier,
        "perf_msr": qsport.portfolio_perf(w_msr, mu, cov),
        "perf_mv": qsport.portfolio_perf(w_mv, mu, cov),
        "prices": prices,
    }


def pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"


def usd(x: float, digits: int = 2) -> str:
    return f"${x:,.{digits}f}"


# ---------------------------------------------------------------------------
# Chrome
# ---------------------------------------------------------------------------

render_header(ENGINE_LABEL, ENGINE_NATIVE)


@st.dialog("📚 Models & Methodology", width="large")
def docs_dialog() -> None:
    render_docs()


with st.sidebar:
    st.markdown("### ◈ CONTROL DECK")
    ticker = st.text_input("Primary ticker", "AAPL", max_chars=12).strip().upper()
    years = st.slider("History window (years)", 2, 10, 5)
    rf = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.045,
                         step=0.005, format="%.3f")
    st.divider()
    st.markdown("### ⚙ SETTINGS")
    if st.button("📚 Documentation · all formulas", use_container_width=True):
        docs_dialog()
    if st.button("🔄 Force data refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.markdown(
        '<p class="qs-note">© 2026 <b>Ismael LADJOHOUNLOU</b><br>'
        'Released under <a href="https://www.gnu.org/licenses/agpl-3.0.html" '
        'style="color:#00E5FF">AGPL-3.0</a> — any modification must be published '
        'under the same terms. Commercial licensing: contact the author.</p>',
        unsafe_allow_html=True,
    )

if not ticker:
    st.info("Enter a ticker in the control deck to boot the pipeline.")
    st.stop()

(tab_data, tab_kalman, tab_vol, tab_pricing, tab_port, tab_bt, tab_opt,
 tab_fc) = st.tabs(
    ["📡 MARKET DATA", "🛰 KALMAN FILTER", "🌋 VOL SURFACE", "⚛ PRICING LAB", "🧭 PORTFOLIO",
     "📈 BACKTEST", "🎯 OPTIMIZER", "🔮 FORECAST"]
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_garch(ticker_: str, years_: int) -> dict:
    lr = qsdata.log_returns(cached_history(ticker_, years_)["Close"])
    return qsmodels.fit_garch11(lr)

# ---------------------------------------------------------------------------
# STAGE 1 — Market data
# ---------------------------------------------------------------------------

with tab_data:
    try:
        with st.spinner(f"Ingesting {ticker} history…"):
            hist = cached_history(ticker, years)
        stats = qsdata.compute_stats(hist["Close"])

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Last close", usd(stats.last_price),
                  pct((stats.last_price - stats.prev_close) / stats.prev_close))
        m2.metric("Drift μ (ann.)", pct(stats.mu_annual))
        m3.metric("Volatility σ (ann.)", pct(stats.sigma_annual))
        m4.metric("Skewness", f"{stats.skew:.3f}")
        m5.metric("Excess kurtosis", f"{stats.kurtosis:.2f}")
        m6.metric("VaR 95% (1d)", pct(stats.var_95_daily))

        rvol = qsdata.rolling_volatility(hist["Close"], 21)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.58, 0.14, 0.28], vertical_spacing=0.03)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Close",
                                 line=dict(color=C.BLUE, width=1.6)), 1, 1)
        fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume",
                             marker=dict(color=C.GRID), showlegend=False), 2, 1)
        fig.add_trace(go.Scatter(x=rvol.index, y=rvol, name="Realized vol (21d, ann.)",
                                 line=dict(color=C.AMBER, width=1.4)), 3, 1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=2, col=1)
        fig.update_yaxes(title_text="σ", tickformat=".0%", row=3, col=1)
        st.plotly_chart(style_fig(fig, height=560, title=f"{ticker} · {years}Y adjusted history"),
                        use_container_width=True)

        c1, c2 = st.columns([0.55, 0.45])
        with c1:
            lr = qsdata.log_returns(hist["Close"])
            xs = np.linspace(float(lr.min()), float(lr.max()), 300)
            gauss = norm.pdf(xs, float(lr.mean()), float(lr.std(ddof=1)))
            figh = go.Figure()
            figh.add_trace(go.Histogram(x=lr, histnorm="probability density", nbinsx=90,
                                        name="Log-returns",
                                        marker=dict(color="rgba(0,229,255,0.45)",
                                                    line=dict(color=C.BLUE, width=0.4))))
            figh.add_trace(go.Scatter(x=xs, y=gauss, name="Gaussian fit",
                                      line=dict(color=C.WHITE, width=1.4, dash="dash")))
            figh.add_vline(x=-stats.var_95_daily, line=dict(color=C.RED, width=1.5, dash="dot"),
                           annotation_text="VaR 95%", annotation_font_color=C.RED)
            st.plotly_chart(style_fig(figh, height=360,
                                      title="Daily log-return distribution vs Gaussian"),
                            use_container_width=True)
            st.markdown(
                f'<p class="qs-note">Fat tails: excess kurtosis {stats.kurtosis:.2f} '
                f'(Gaussian = 0). This is why the pricing lab carries jump models.</p>',
                unsafe_allow_html=True)
        with c2:
            ic1, ic2 = st.columns(2)
            intra_iv = ic1.selectbox("Intraday interval", ["1m", "5m", "15m"], key="intra_iv")
            intra_pd = ic2.selectbox("Window", ["1d", "5d"], key="intra_pd")
            try:
                intra = cached_intraday(ticker, intra_pd, intra_iv)
                px = intra["Close"]
                vol_i = intra["Volume"].replace(0.0, np.nan)
                day_key = pd.Series(intra.index.date, index=intra.index)
                vwap = ((px * vol_i).groupby(day_key).cumsum()
                        / vol_i.groupby(day_key).cumsum())
                last_px = float(px.iloc[-1])
                last_ts = px.index[-1]

                figi = go.Figure()
                figi.add_trace(go.Scatter(x=px.index, y=px, name=f"{intra_iv} close",
                                          line=dict(color=C.GREEN, width=1.3)))
                figi.add_trace(go.Scatter(x=vwap.index, y=vwap, name="Session VWAP",
                                          line=dict(color=C.AMBER, width=1.1, dash="dot")))
                figi.add_hline(y=last_px, line=dict(color=C.WHITE, width=0.8, dash="dot"),
                               annotation_text=f"{last_px:,.2f}",
                               annotation_font_color=C.WHITE,
                               annotation_position="right")
                breaks = [dict(bounds=["sat", "mon"])]
                if str(px.index.tz) == "America/New_York":
                    breaks.append(dict(bounds=[16, 9.5], pattern="hour"))
                figi.update_xaxes(rangebreaks=breaks)
                st.plotly_chart(
                    style_fig(figi, height=360,
                              title=f"Intraday · {intra_iv} bars · last "
                                    f"{last_ts.strftime('%d %b %H:%M')} (60s cache)"),
                    use_container_width=True)
                st.markdown(
                    '<p class="qs-note">Non-trading hours and weekends are removed from '
                    'the axis. VWAP resets each session. Yahoo intraday can lag a few '
                    'minutes and shows the last completed session when the market is '
                    'closed.</p>', unsafe_allow_html=True)
            except ValueError as exc:
                st.info(f"Intraday feed unavailable: {exc}")
    except ValueError as exc:
        st.error(f"Stage 1 failed for '{ticker}': {exc}")
        st.stop()

# ---------------------------------------------------------------------------
# STAGE 2 — Kalman filter
# ---------------------------------------------------------------------------

with tab_kalman:
    k1, k2, k3 = st.columns([0.3, 0.35, 0.35])
    source = k1.radio("Signal source", ["Daily close", "Intraday 1-minute"], horizontal=False)
    model = k2.radio("State model", ["Local level", "Level + drift"], horizontal=False,
                     help="Level + drift tracks an unobserved trend (alpha) as a second state")
    resp = k3.slider("Responsiveness log₁₀(Q/R)", -6.0, 0.0, -2.5, 0.1,
                     help="Higher = filter trusts new observations more (faster, noisier)")
    proj_h = k3.slider("Prediction horizon (bars ahead)", 0, 60, 20, 5,
                       help="h-step-ahead state projection x̂(t+h|t) = Aʰ x̂(t|t) with "
                            "uncertainty compounding through Q")

    try:
        if source == "Daily close":
            series = cached_history(ticker, years)["Close"].iloc[-756:]
        else:
            series = cached_intraday(ticker, "5d", "1m")["Close"]
        y = series.to_numpy(dtype=float)
        # Observation noise: half the variance of first differences is the
        # classical moment estimate when the state is a slow random walk.
        r_hat = max(0.5 * float(np.var(np.diff(y))), 1e-10)
        q_hat = r_hat * 10.0 ** resp
        model_key = "local_level" if model == "Local level" else "local_trend"
        res = core.kalman(y, model=model_key, q=q_hat, r=r_hat, q_drift=q_hat * 1e-3)

        level = np.asarray(res["x_filt"])[:, 0]
        smooth = np.asarray(res["x_smooth"])[:, 0]
        p_diag = np.asarray(res["P_filt"])[:, 0]
        band = 2.0 * np.sqrt(np.maximum(p_diag, 0.0))
        innov = np.asarray(res["innovations"])
        innov_sd = np.sqrt(np.asarray(res["innov_var"]))
        z = innov[5:] / innov_sd[5:]  # drop filter burn-in

        noise_var = float(np.var(y - level))
        raw_var = float(np.var(np.diff(y)))
        km1, km2, km3, km4 = st.columns(4)
        km1.metric("Log-likelihood", f"{res['loglik']:.1f}")
        km2.metric("Extracted noise σ", f"{np.sqrt(noise_var):.4f}")
        km3.metric("Innovation z-score μ", f"{float(np.mean(z)):.3f}")
        km4.metric("Innovation z-score σ", f"{float(np.std(z)):.3f}",
                   help="≈1.0 means the filter's uncertainty is well calibrated")

        figk = go.Figure()
        figk.add_trace(go.Scatter(x=series.index, y=y, name="Raw market price", mode="lines",
                                  line=dict(color=C.MUTED, width=0.9), opacity=0.65))
        figk.add_trace(go.Scatter(x=series.index, y=level + band, showlegend=False,
                                  line=dict(width=0), hoverinfo="skip"))
        figk.add_trace(go.Scatter(x=series.index, y=level - band, name="±2σ filter band",
                                  fill="tonexty", fillcolor="rgba(255,234,0,0.10)",
                                  line=dict(width=0), hoverinfo="skip"))
        figk.add_trace(go.Scatter(x=series.index, y=level, name="Kalman filtered",
                                  line=dict(color=C.AMBER, width=1.7)))
        figk.add_trace(go.Scatter(x=series.index, y=smooth, name="RTS smoothed",
                                  line=dict(color=C.BLUE, width=1.2, dash="dot")))

        if proj_h > 0:
            # h-step-ahead prediction from the last filtered state:
            # x(t+h|t) = A^h x(t|t),  P grows by A P A' + Q each step.
            dim = int(res["dim"])
            if dim == 1:
                A_m = np.array([[1.0]])
                Q_m = np.array([[q_hat]])
            else:
                A_m = np.array([[1.0, 1.0], [0.0, 1.0]])
                Q_m = np.diag([q_hat, q_hat * 1e-3])
            x_p = np.asarray(res["x_filt"])[-1].copy()
            P_p = np.asarray(res["P_filt"])[-1].reshape(dim, dim).copy()
            proj_mean = np.empty(proj_h)
            proj_sd = np.empty(proj_h)
            for h in range(proj_h):
                x_p = A_m @ x_p
                P_p = A_m @ P_p @ A_m.T + Q_m
                proj_mean[h] = x_p[0]
                proj_sd[h] = np.sqrt(max(P_p[0, 0], 0.0))
            if source == "Daily close":
                fut_x = pd.bdate_range(series.index[-1], periods=proj_h + 1)[1:]
            else:
                step = (series.index[1:] - series.index[:-1]).median()
                fut_x = pd.DatetimeIndex([series.index[-1] + step * (h + 1)
                                          for h in range(proj_h)])
            figk.add_trace(go.Scatter(x=fut_x, y=proj_mean + 2 * proj_sd, showlegend=False,
                                      line=dict(width=0), hoverinfo="skip"))
            figk.add_trace(go.Scatter(x=fut_x, y=proj_mean - 2 * proj_sd,
                                      name="Prediction ±2σ", fill="tonexty",
                                      fillcolor="rgba(0,229,255,0.12)",
                                      line=dict(width=0), hoverinfo="skip"))
            figk.add_trace(go.Scatter(x=fut_x, y=proj_mean, name=f"Prediction +{proj_h} bars",
                                      line=dict(color=C.BLUE, width=1.8, dash="dash")))
            figk.add_vline(x=series.index[-1], line=dict(color=C.MUTED, width=1, dash="dot"))
            figk.add_annotation(x=fut_x[-1], y=float(proj_mean[-1]),
                                text=f"{proj_mean[-1]:,.2f}", showarrow=False,
                                xanchor="left", font=dict(color=C.BLUE, size=11))

        st.plotly_chart(style_fig(figk, height=460,
                                  title=f"{ticker} · latent state extraction ({model.lower()})"),
                        use_container_width=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            xs = np.linspace(-4, 4, 200)
            figz = go.Figure()
            figz.add_trace(go.Histogram(x=z, histnorm="probability density", nbinsx=60,
                                        name="Standardized innovations",
                                        marker=dict(color="rgba(255,234,0,0.4)",
                                                    line=dict(color=C.AMBER, width=0.4))))
            figz.add_trace(go.Scatter(x=xs, y=norm.pdf(xs), name="N(0,1)",
                                      line=dict(color=C.WHITE, width=1.3, dash="dash")))
            st.plotly_chart(style_fig(figz, height=330, title="Innovation whiteness check"),
                            use_container_width=True)
        with cc2:
            if model_key == "local_trend":
                drift = np.asarray(res["x_filt"])[:, 1]
                figd = go.Figure()
                figd.add_trace(go.Scatter(x=series.index, y=drift, name="Filtered drift/step",
                                          line=dict(color=C.GREEN, width=1.4),
                                          fill="tozeroy", fillcolor="rgba(0,230,118,0.07)"))
                figd.add_hline(y=0, line=dict(color=C.MUTED, width=1, dash="dot"))
                st.plotly_chart(style_fig(figd, height=330,
                                          title="Latent drift state (alpha tracking)"),
                                use_container_width=True)
            else:
                resid = y - level
                figr = go.Figure()
                figr.add_trace(go.Scatter(x=series.index, y=resid, mode="lines",
                                          name="Extracted noise", line=dict(color=C.MUTED, width=0.8)))
                figr.add_hline(y=0, line=dict(color=C.AMBER, width=1, dash="dot"))
                st.plotly_chart(style_fig(figr, height=330,
                                          title="Extracted measurement noise (price − state)"),
                                use_container_width=True)
        st.markdown(
            f'<p class="qs-note">Q/R = 10^{resp:.1f} · R̂ = {r_hat:.3g} (moment estimate) · '
            f'raw Δy variance {raw_var:.3g}. The filter runs in the '
            f'{"C++ core" if ENGINE_NATIVE else "NumPy fallback"} with an RTS smoothing pass.</p>',
            unsafe_allow_html=True)
    except ValueError as exc:
        st.error(f"Stage 2 failed: {exc}")

# ---------------------------------------------------------------------------
# STAGE 3 — Volatility surface
# ---------------------------------------------------------------------------

with tab_vol:
    try:
        with st.spinner("Pulling live option chain…"):
            chain, spot = cached_chain(ticker)
        ivs = np.asarray(core.implied_vol(
            chain["mid"].to_numpy(), float(spot), chain["strike"].to_numpy(),
            chain["T"].to_numpy(), float(rf), chain["is_call"].to_numpy(bool)))
        chain = chain.assign(iv=ivs).dropna(subset=["iv"])
        chain = chain[(chain["iv"] > 0.01) & (chain["iv"] < 3.0)]
        if len(chain) < 12:
            raise ValueError("Too few valid implied vols after filtering")

        # OTM quotes carry the cleanest smile information.
        otm = chain[((chain["is_call"]) & (chain["strike"] >= spot))
                    | ((~chain["is_call"]) & (chain["strike"] < spot))]
        nearest_T = float(otm["T"].min())
        atm_row = otm.loc[(otm["T"] == nearest_T)]
        atm_iv = float(atm_row.iloc[(atm_row["strike"] - spot).abs().argsort()[:1]]["iv"].iloc[0])
        rmse_flat = float(np.sqrt(np.mean((otm["iv"] - atm_iv) ** 2)))

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Spot", usd(spot))
        v2.metric("ATM IV (front expiry)", pct(atm_iv))
        v3.metric("Valid quotes", f"{len(chain):,}")
        v4.metric("Flat-vol RMSE", pct(rmse_flat),
                  help="Pricing error of a single flat volatility vs the observed smile — "
                       "this is the smile the Black-Scholes model cannot see")

        col3d, colsmile = st.columns([0.58, 0.42])
        with col3d:
            pts_k = otm["strike"].to_numpy()
            pts_t = otm["T"].to_numpy()
            pts_v = otm["iv"].to_numpy()
            kg = np.linspace(pts_k.min(), pts_k.max(), 45)
            tg = np.linspace(pts_t.min(), pts_t.max(), 45)
            KG, TG = np.meshgrid(kg, tg)
            IVG = griddata((pts_k, pts_t), pts_v, (KG, TG), method="linear")
            fig3d = go.Figure()
            fig3d.add_trace(go.Surface(x=KG, y=TG, z=IVG, colorscale=PURPLE_SCALE,
                                       opacity=0.92, name="IV surface",
                                       colorbar=dict(title="IV", tickformat=".0%", len=0.6)))
            fig3d.add_trace(go.Scatter3d(x=pts_k, y=pts_t, z=pts_v, mode="markers",
                                         name="Market quotes",
                                         marker=dict(size=2.4, color=C.BLUE, opacity=0.75)))
            fig3d.update_layout(scene=dict(
                xaxis=dict(title="Strike", backgroundcolor=C.BG, gridcolor=C.GRID),
                yaxis=dict(title="Maturity (y)", backgroundcolor=C.BG, gridcolor=C.GRID),
                zaxis=dict(title="Implied vol", tickformat=".0%",
                           backgroundcolor=C.BG, gridcolor=C.GRID),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.65)),
            ))
            st.plotly_chart(style_fig(fig3d, height=560,
                                      title=f"{ticker} · live implied volatility surface (OTM)"),
                            use_container_width=True)
        with colsmile:
            expiries = sorted(otm["expiry"].unique())
            sel = st.multiselect("Expiries", expiries, default=expiries[:4])
            figs = go.Figure()
            palette = [C.BLUE, C.GREEN, C.AMBER, C.PURPLE, C.WHITE, C.RED]
            for i, exp in enumerate(sel):
                sub = otm[otm["expiry"] == exp].sort_values("strike")
                figs.add_trace(go.Scatter(x=sub["strike"] / spot, y=sub["iv"],
                                          name=str(exp), mode="lines+markers",
                                          marker=dict(size=4),
                                          line=dict(color=palette[i % len(palette)], width=1.4)))
            figs.add_vline(x=1.0, line=dict(color=C.MUTED, width=1, dash="dot"),
                           annotation_text="ATM", annotation_font_color=C.MUTED)
            figs.add_hline(y=atm_iv, line=dict(color=C.RED, width=1, dash="dot"),
                           annotation_text="flat BS vol", annotation_font_color=C.RED)
            figs.update_xaxes(title="Moneyness K/S", tickformat=".2f")
            figs.update_yaxes(title="Implied vol", tickformat=".0%")
            st.plotly_chart(style_fig(figs, height=490, title="Smile by expiry"),
                            use_container_width=True)
        st.markdown(
            '<p class="qs-note">IV backed out per quote via Newton-Raphson with bisection '
            'fallback in the native engine; quotes outside the no-arbitrage band are '
            'rejected (NaN). Mid = (bid+ask)/2 when a two-sided market exists.</p>',
            unsafe_allow_html=True)
    except ValueError as exc:
        st.error(f"Stage 3 failed: {exc}")
        st.info("Options data requires a US-listed underlying with a liquid chain "
                "(e.g. AAPL, SPY, TSLA, NVDA).")

# ---------------------------------------------------------------------------
# STAGE 4 — Pricing lab
# ---------------------------------------------------------------------------

with tab_pricing:
    try:
        hist = cached_history(ticker, years)
        stats = qsdata.compute_stats(hist["Close"])
        S0 = stats.last_price
        sig0 = max(stats.sigma_annual, 0.01)
    except ValueError:
        S0, sig0 = 100.0, 0.2

    with st.form("pricing_form"):
        p1, p2, p3, p4 = st.columns(4)
        model_name = p1.selectbox("Dynamics", ["GBM (Black-Scholes)", "Heston (stoch. vol)",
                                               "Merton (jump-diffusion)", "Variance Gamma"])
        kind = p2.selectbox("Payoff", ["Call", "Put"])
        K = p3.number_input("Strike", min_value=0.01, value=float(round(S0)), step=1.0)
        T = p4.number_input("Maturity (years)", min_value=0.05, max_value=5.0, value=1.0,
                            step=0.05)
        p5, p6, p7, p8 = st.columns(4)
        sigma_in = p5.number_input("Volatility σ", min_value=0.01, max_value=3.0,
                                   value=float(round(sig0, 3)), step=0.01, format="%.3f")
        n_paths = p6.select_slider("MC paths", [10_000, 25_000, 50_000, 100_000, 200_000],
                                   value=100_000)
        exercise = p7.selectbox("Exercise (PDE)", ["European", "American"])
        barrier_type = p8.selectbox("Barrier", ["none", "up-and-out", "down-and-out",
                                                "up-and-in", "down-and-in"])
        barrier_level = st.slider("Barrier level (× spot)", 0.5, 2.0,
                                  1.25 if "up" in barrier_type else 0.8, 0.01,
                                  disabled=(barrier_type == "none"))
        with st.expander("Model parameters (Heston / Merton / Variance Gamma)"):
            h1, h2, h3, h4, h5 = st.columns(5)
            v0 = h1.number_input("Heston v₀", 0.001, 2.0, float(round(sig0**2, 4)), format="%.4f")
            kappa = h2.number_input("κ mean-rev", 0.01, 20.0, 1.5)
            theta_h = h3.number_input("θ long-run var", 0.001, 2.0, float(round(sig0**2, 4)),
                                      format="%.4f")
            xi = h4.number_input("ξ vol-of-vol", 0.01, 3.0, 0.5)
            rho = h5.number_input("ρ correlation", -0.999, 0.999, -0.7)
            j1, j2, j3, j4, j5 = st.columns(5)
            lam = j1.number_input("λ jump intensity", 0.0, 10.0, 0.5)
            mu_j = j2.number_input("μⱼ jump mean", -1.0, 1.0, -0.05)
            sig_j = j3.number_input("σⱼ jump vol", 0.0, 1.0, 0.15)
            vg_theta = j4.number_input("VG θ", -1.0, 1.0, -0.1)
            vg_nu = j5.number_input("VG ν", 0.01, 2.0, 0.2)
        submitted = st.form_submit_button("⚡ RUN PRICING ENGINES", use_container_width=True)

    if submitted:
        mkey = {"GBM (Black-Scholes)": "gbm", "Heston (stoch. vol)": "heston",
                "Merton (jump-diffusion)": "merton", "Variance Gamma": "vg"}[model_name]
        is_call = kind == "Call"
        kw = dict(S0=float(S0), r=float(rf), T=float(T), sigma=float(sigma_in),
                  v0=float(v0), kappa=float(kappa), theta=float(theta_h), xi=float(xi),
                  rho=float(rho), lam=float(lam), mu_j=float(mu_j), sig_j=float(sig_j),
                  vg_theta=float(vg_theta), vg_nu=float(vg_nu))
        steps = max(int(252 * T), 50)
        try:
            with st.spinner(f"Simulating {n_paths:,} paths + solving PDE grid…"):
                mc = core.mc_price(mkey, float(K), is_call, n_paths=int(n_paths),
                                   steps=steps, seed=42, barrier_type=barrier_type,
                                   barrier=float(barrier_level * S0), **kw)
                mc_vanilla = (core.mc_price(mkey, float(K), is_call, n_paths=int(n_paths),
                                            steps=steps, seed=42, barrier_type="none",
                                            barrier=0.0, **kw)
                              if barrier_type != "none" else None)
                viz = np.asarray(core.simulate_paths(mkey, 250, min(steps, 252), seed=7, **kw))
                pde = core.pde_price(float(S0), float(K), float(rf), float(sigma_in),
                                     float(T), is_call=is_call,
                                     american=(exercise == "American"), Ns=220, Nt=220)
                bs = core.bs_price(float(S0), float(K), float(T), float(rf),
                                   float(sigma_in), is_call)
            st.session_state["pricing"] = dict(
                mc=dict(mc), mc_vanilla=dict(mc_vanilla) if mc_vanilla else None,
                viz=viz, pde=dict(pde), bs=float(bs), mkey=mkey, is_call=is_call,
                K=float(K), T=float(T), S0=float(S0), rf=float(rf),
                barrier_type=barrier_type, barrier=float(barrier_level * S0),
                exercise=exercise, n_paths=int(n_paths), model_name=model_name)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Pricing failed: {exc}")

    P = st.session_state.get("pricing")
    if P:
        term = np.asarray(P["mc"]["terminal"])
        q05, q01 = float(np.quantile(term, 0.05)), float(np.quantile(term, 0.01))

        r1, r2, r3, r4, r5, r6 = st.columns(6)
        label = "Barrier MC" if P["barrier_type"] != "none" else "Monte Carlo"
        r1.metric(f"{label} price", usd(P["mc"]["price"]),
                  f"± {P['mc']['std_error']:.4f} SE", delta_color="off")
        r2.metric(f"PDE {P['exercise']}", usd(P["pde"]["price"]),
                  f"{(P['pde']['price'] - P['bs']):+.4f} vs BS", delta_color="off")
        r3.metric("Black-Scholes closed form", usd(P["bs"]))
        r4.metric("Δ Delta (PDE)", f"{P['pde']['delta']:.4f}")
        r5.metric("Γ Gamma (PDE)", f"{P['pde']['gamma']:.5f}")
        r6.metric("Θ Theta /yr (PDE)", f"{P['pde']['theta']:.4f}")
        if P["mkey"] != "gbm":
            st.markdown(
                f'<p class="qs-note">MC runs under {P["model_name"]}; the PDE and closed-form '
                'columns are the Black-Scholes benchmark at the same σ — the gap is the '
                'model premium (jumps / stochastic vol).</p>', unsafe_allow_html=True)
        if P["mc_vanilla"]:
            ko = 1.0 - P["mc"]["price"] / max(P["mc_vanilla"]["price"], 1e-12)
            st.markdown(
                f'<p class="qs-note">Vanilla equivalent: {usd(P["mc_vanilla"]["price"])} — the '
                f'{P["barrier_type"]} barrier at {usd(P["barrier"])} removes '
                f'{pct(max(ko, 0.0))} of vanilla value.</p>', unsafe_allow_html=True)

        viz = P["viz"]
        tgrid = np.linspace(0.0, P["T"], viz.shape[1])
        figp = make_subplots(rows=1, cols=2, shared_yaxes=True,
                             column_widths=[0.74, 0.26], horizontal_spacing=0.015)
        for row in viz[:250]:
            figp.add_trace(go.Scatter(x=tgrid, y=row, mode="lines", showlegend=False,
                                      line=dict(color="rgba(0,229,255,0.05)", width=0.7),
                                      hoverinfo="skip"), 1, 1)
        figp.add_trace(go.Scatter(x=tgrid, y=viz.mean(axis=0), name="Mean path",
                                  line=dict(color=C.WHITE, width=1.8)), 1, 1)
        figp.add_trace(go.Histogram(y=term, nbinsy=120, name="Terminal S(T)",
                                    marker=dict(color="rgba(0,229,255,0.5)")), 1, 2)
        for level, color, name in ((P["K"], C.GREEN, "Strike"),
                                   (q05, C.AMBER, "VaR 95%"),
                                   (q01, C.RED, "VaR 99%")):
            figp.add_hline(y=level, line=dict(color=color, width=1.2, dash="dot"),
                           annotation_text=name, annotation_font_color=color, row=1, col=1)
            figp.add_hline(y=level, line=dict(color=color, width=1.2, dash="dot"), row=1, col=2)
        if P["barrier_type"] != "none":
            figp.add_hline(y=P["barrier"], line=dict(color=C.PURPLE, width=1.4),
                           annotation_text="Barrier", annotation_font_color=C.PURPLE,
                           row=1, col=1)
        figp.update_xaxes(title="t (years)", row=1, col=1)
        figp.update_xaxes(showticklabels=False, row=1, col=2)
        st.plotly_chart(
            style_fig(figp, height=470,
                      title=f"{P['model_name']} · 250 of {P['n_paths']:,} risk-neutral paths "
                            f"+ terminal distribution"),
            use_container_width=True)
        st.markdown(
            f'<p class="qs-note">Terminal quantiles: 5% → {usd(q05)} '
            f'({pct(q05 / P["S0"] - 1)}) · 1% → {usd(q01)} ({pct(q01 / P["S0"] - 1)}). '
            f'Priced on {P["n_paths"]:,} paths in the '
            f'{"C++ engine (multithreaded)" if ENGINE_NATIVE else "NumPy fallback"}.</p>',
            unsafe_allow_html=True)

        pde = P["pde"]
        s_grid = np.asarray(pde["s_grid"])
        t_grid = np.asarray(pde["t_grid"])
        surf = np.asarray(pde["surface"])
        s_keep = s_grid <= 2.2 * max(P["S0"], P["K"])
        t_stride = max(1, len(t_grid) // 160)
        figh = go.Figure(go.Heatmap(
            x=t_grid[::t_stride], y=s_grid[s_keep], z=surf[::t_stride, :][:, s_keep].T,
            colorscale=PURPLE_SCALE, colorbar=dict(title="V(S,t)")))
        figh.add_hline(y=P["S0"], line=dict(color=C.BLUE, width=1.3, dash="dot"),
                       annotation_text="Spot", annotation_font_color=C.BLUE)
        figh.add_hline(y=P["K"], line=dict(color=C.GREEN, width=1.3, dash="dot"),
                       annotation_text="Strike", annotation_font_color=C.GREEN)
        figh.update_xaxes(title="Calendar time t (years)")
        figh.update_yaxes(title="Asset price S")
        st.plotly_chart(
            style_fig(figh, height=450,
                      title=f"Crank-Nicolson value grid · {P['exercise']} {'call' if P['is_call'] else 'put'} "
                            f"({'PSOR' if P['exercise'] == 'American' else 'Thomas'} solver, "
                            f"220×220 nodes)"),
            use_container_width=True)
    else:
        st.markdown('<p class="qs-note">Configure the instrument and hit '
                    '⚡ RUN PRICING ENGINES.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# STAGE 5 — Portfolio optimization
# ---------------------------------------------------------------------------

with tab_port:
    default_basket = "AAPL, MSFT, GOOGL, AMZN, NVDA, TLT, GLD"
    basket_raw = st.text_input("Basket (comma-separated tickers)", default_basket)
    tickers = tuple(dict.fromkeys(t.strip().upper() for t in basket_raw.split(",") if t.strip()))
    if len(tickers) < 2:
        st.warning("Enter at least two tickers.")
    else:
        try:
            with st.spinner("Optimizing basket…"):
                sol = cached_portfolio(tickers, years, float(rf))
            (ret_msr, vol_msr), (ret_mv, vol_mv) = sol["perf_msr"], sol["perf_mv"]
            sh_msr = (ret_msr - rf) / vol_msr

            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Max-Sharpe return", pct(ret_msr))
            b2.metric("Max-Sharpe vol", pct(vol_msr))
            b3.metric("Sharpe ratio", f"{sh_msr:.3f}")
            b4.metric("Min-var return", pct(ret_mv))
            b5.metric("Min-var vol", pct(vol_mv))

            cloud = sol["cloud"]
            figf = go.Figure()
            figf.add_trace(go.Scatter(
                x=cloud["vols"], y=cloud["returns"], mode="markers", name="Random portfolios",
                marker=dict(size=4, color=cloud["sharpes"], colorscale=BLUE_GREEN_SCALE,
                            opacity=0.55, colorbar=dict(title="Sharpe", len=0.7)),
                hovertemplate="σ %{x:.1%} · μ %{y:.1%}<extra></extra>"))
            fr = sol["frontier"]
            figf.add_trace(go.Scatter(x=fr["vols"], y=fr["returns"], name="Efficient frontier",
                                      line=dict(color=C.WHITE, width=2)))
            asset_vols = np.sqrt(np.diag(sol["cov"]))
            figf.add_trace(go.Scatter(x=asset_vols, y=sol["mu"], mode="markers+text",
                                      name="Assets", text=list(sol["tickers"]),
                                      textposition="top center",
                                      textfont=dict(size=10, color=C.MUTED),
                                      marker=dict(symbol="x", size=9, color=C.MUTED)))
            figf.add_trace(go.Scatter(x=[vol_msr], y=[ret_msr], mode="markers",
                                      name="★ Max Sharpe (tangent)",
                                      marker=dict(symbol="star", size=17, color=C.GREEN,
                                                  line=dict(color=C.WHITE, width=1))))
            figf.add_trace(go.Scatter(x=[vol_mv], y=[ret_mv], mode="markers",
                                      name="◆ Min variance",
                                      marker=dict(symbol="diamond", size=13, color=C.AMBER,
                                                  line=dict(color=C.WHITE, width=1))))
            # Capital market line through the tangent portfolio.
            cml_x = np.array([0.0, float(cloud["vols"].max()) * 1.05])
            figf.add_trace(go.Scatter(x=cml_x, y=rf + sh_msr * cml_x, name="Capital market line",
                                      line=dict(color=C.GREEN, width=1, dash="dot")))
            figf.update_xaxes(title="Annualized volatility σ", tickformat=".0%", rangemode="tozero")
            figf.update_yaxes(title="Annualized return μ", tickformat=".0%")
            st.plotly_chart(style_fig(figf, height=540,
                                      title=f"Markowitz frontier · {len(tickers)} assets · "
                                            f"4,000 sampled portfolios · {years}Y window"),
                            use_container_width=True)

            wc1, wc2 = st.columns(2)
            for col, w, name, color in ((wc1, sol["w_msr"], "Max-Sharpe weights", C.GREEN),
                                        (wc2, sol["w_mv"], "Min-variance weights", C.AMBER)):
                order = np.argsort(w)
                figw = go.Figure(go.Bar(
                    x=w[order], y=[sol["tickers"][i] for i in order], orientation="h",
                    marker=dict(color=color, opacity=0.75), text=[pct(x, 1) for x in w[order]],
                    textposition="outside", textfont=dict(size=11)))
                figw.update_xaxes(tickformat=".0%", range=[0, max(0.35, float(w.max()) * 1.25)])
                with col:
                    st.plotly_chart(style_fig(figw, height=320, title=name),
                                    use_container_width=True)
            st.markdown("##### What these weights actually did — realized performance")
            rets_p = sol["prices"].pct_change().dropna()
            w_eq = np.full(len(tickers), 1.0 / len(tickers))
            figpc = go.Figure()
            perf_rows = []
            for w_vec, name, color in ((sol["w_msr"], "Max Sharpe", C.GREEN),
                                       (sol["w_mv"], "Min variance", C.AMBER),
                                       (w_eq, "Equal weight", C.BLUE)):
                pr = rets_p @ w_vec
                eqc = (1.0 + pr).cumprod()
                figpc.add_trace(go.Scatter(x=eqc.index, y=eqc, name=name,
                                           line=dict(color=color, width=1.5)))
                pm = qsbt.perf_metrics(pr, float(rf))
                perf_rows.append((name, pm))
            figpc.update_yaxes(title="Growth of $1", type="log")
            st.plotly_chart(
                style_fig(figpc, height=400,
                          title=f"Daily-rebalanced to fixed weights · {years}Y realized "
                                "history (in-sample: weights were fit on this window)"),
                use_container_width=True)
            pf_cols = st.columns(3)
            for col, (name, pm) in zip(pf_cols, perf_rows):
                col.metric(f"{name} · CAGR", pct(pm["cagr"]),
                           f"Sharpe {pm['sharpe']:.2f} · DD {pct(pm['max_drawdown'])}",
                           delta_color="off")

            st.markdown("##### Deploy it — allocation for your capital")
            cap_p = st.number_input("Capital to allocate ($)", 1000.0, 1e9, 100_000.0,
                                    1000.0, key="pf_cap")
            last_px = sol["prices"].iloc[-1]
            alloc = pd.DataFrame({
                "Ticker": list(sol["tickers"]),
                "Weight": sol["w_msr"],
                "Allocation ($)": sol["w_msr"] * cap_p,
                "Last price": last_px.to_numpy(),
                "Shares": np.floor(sol["w_msr"] * cap_p / last_px.to_numpy()),
            })
            st.dataframe(
                alloc.style.format({"Weight": "{:.1%}", "Allocation ($)": "${:,.0f}",
                                    "Last price": "${:,.2f}", "Shares": "{:,.0f}"}),
                use_container_width=True, hide_index=True)

            st.markdown(
                '<p class="qs-note">Long-only SLSQP with ridge-regularized covariance; '
                'frontier solved as 40 constrained min-variance programs. Sampling cloud: '
                'Dirichlet(1) — uniform on the simplex. Realized curves are in-sample '
                '(weights estimated on the same window) — treat them as capability '
                'evidence, not a promise; see ⚙ Settings → Documentation §10.</p>',
                unsafe_allow_html=True)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Stage 5 failed: {exc}")

# ---------------------------------------------------------------------------
# STAGE 6 — Backtesting
# ---------------------------------------------------------------------------

with tab_bt:
    bc1, bc2, bc3, bc4 = st.columns(4)
    timeframe = bc1.selectbox("Timeframe", ["1d", "1h", "30m", "15m", "5m"], index=0,
                              key="bt_tf",
                              help="Yahoo lookback caps: 1h → 2 years, 30m/15m/5m → 60 days")
    capital = bc2.number_input("Capital ($)", 100.0, 1e9, 10_000.0, step=1000.0,
                               key="bt_capital")
    lot = bc3.number_input("Position size × (lot / leverage)", 0.1, 10.0, 1.0, 0.1,
                           key="bt_lot",
                           help="Scales every position; costs scale with it too")
    cost_bps = bc4.number_input("Cost per trade (bps)", 0.0, 50.0, 5.0, 0.5, key="bt_cost")

    try:
        bars_bt = cached_bars(ticker, timeframe, years)
        close_bt, volume_bt = bars_bt["Close"], bars_bt["Volume"]
    except ValueError as exc:
        st.error(f"Stage 6 failed: {exc}")
        close_bt = None

    if close_bt is not None:
        ppy = qsbt.PERIODS_PER_YEAR[timeframe]
        s1, s2 = st.columns([0.45, 0.55])
        strat_name = s1.selectbox("Strategy", list(qsbt.STRATEGIES), key="bt_strat")
        shorts_allowed = strat_name not in ("Volatility targeting", "Buy & hold")
        long_short = s2.toggle("Allow short positions (−1 exposure on sell signals)",
                               value=False, key="bt_short", disabled=not shorts_allowed)
        long_short = bool(long_short) and shorts_allowed

        pcol = st.columns(3)
        if strat_name == "MA crossover":
            ma_type = pcol[0].selectbox("MA type", ["SMA", "EMA"], key="bt_matype")
            fast = pcol[1].slider("Fast window", 5, 100, 20, key="bt_fast")
            slow = pcol[2].slider("Slow window", 20, 300, 50, key="bt_slow")
            signal = qsbt.signal_ma_cross(close_bt, fast, max(slow, fast + 1),
                                          long_short, ma_type)
        elif strat_name == "Kalman trend (drift sign)":
            resp_bt = pcol[0].slider("Filter responsiveness log₁₀(Q/R)", -6.0, -1.0, -3.0,
                                     0.1, key="bt_resp")
            signal = qsbt.signal_kalman_trend(close_bt, resp_bt, long_short)
        elif strat_name == "Mean reversion (z-score)":
            win = pcol[0].slider("Lookback window", 5, 60, 20, key="bt_zwin")
            entry = pcol[1].slider("Entry threshold |z|", 0.5, 3.0, 1.5, 0.1, key="bt_zentry")
            signal = qsbt.signal_zscore_meanrev(close_bt, win, entry, long_short)
        elif strat_name == "Volatility targeting":
            tv = pcol[0].slider("Target volatility", 0.05, 0.40, 0.15, 0.01, key="bt_tv")
            win = pcol[1].slider("Realized vol window (bars)", 10, 63, 21, key="bt_vwin")
            signal = qsbt.signal_vol_target(close_bt, tv, win, periods_per_year=ppy)
        else:
            signal = qsbt.signal_buy_hold(close_bt)

        with st.expander("🔧 Position filters (composable, causal)"):
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                use_trend = st.toggle("Trend regime filter", key="bt_ftrend",
                                      help="Longs only above the trend MA, shorts only below")
                trend_win = st.slider("Trend MA window (bars)", 50, 400, 200, 10,
                                      key="bt_ftw", disabled=not use_trend)
                trend_ma = st.selectbox("Trend MA type", ["SMA", "EMA"], key="bt_ftma",
                                        disabled=not use_trend)
            with fcol2:
                use_volf = st.toggle("Volume confirmation filter", key="bt_fvol",
                                     help="Hold positions only when volume ≥ multiple of "
                                          "its 20-bar average")
                vol_mult = st.slider("Volume ≥ × average(20)", 0.5, 3.0, 1.0, 0.1,
                                     key="bt_fvm", disabled=not use_volf)
        if use_trend:
            signal = qsbt.filter_trend(signal, close_bt, trend_win, trend_ma)
        if use_volf:
            signal = qsbt.filter_volume(signal, volume_bt, 20, vol_mult)

        try:
            bt = qsbt.run_backtest(close_bt, signal, cost_bps=cost_bps, rf=float(rf),
                                   periods_per_year=ppy, capital=float(capital),
                                   leverage=float(lot))
            m, b = bt.metrics, bt.benchmark_metrics

            r1c = st.columns(4)
            r1c[0].metric("Final equity", usd(float(bt.equity_usd.iloc[-1])),
                          f"{bt.pnl_usd:+,.0f} $ P&L")
            r1c[1].metric("CAGR", pct(m["cagr"]), pct(m["cagr"] - b["cagr"]))
            r1c[2].metric("Sharpe", f"{m['sharpe']:.2f}", f"{m['sharpe'] - b['sharpe']:+.2f}")
            r1c[3].metric("Max drawdown", pct(m["max_drawdown"]),
                          pct(m["max_drawdown"] - b["max_drawdown"]))
            r2c = st.columns(5)
            r2c[0].metric("Sortino", f"{m['sortino']:.2f}"
                          if np.isfinite(m["sortino"]) else "—")
            r2c[1].metric("Calmar", f"{m['calmar']:.2f}" if np.isfinite(m["calmar"]) else "—")
            r2c[2].metric("Hit rate", pct(m["hit_rate"], 1) if np.isfinite(m["hit_rate"]) else "—")
            r2c[3].metric("Time in market", pct(bt.exposure, 1))
            r2c[4].metric("Annual turnover", f"{bt.turnover_annual:.1f}×",
                          f"−{bt.turnover_annual * cost_bps / 100:.2f}%/yr in costs",
                          delta_color="off")

            log_scale = st.toggle("Log scale", value=(timeframe == "1d"), key="bt_log")
            fige = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.72, 0.28], vertical_spacing=0.04)
            fige.add_trace(go.Scatter(x=bt.benchmark_equity.index,
                                      y=float(capital) * bt.benchmark_equity,
                                      name=f"Buy & hold {ticker}",
                                      line=dict(color=C.MUTED, width=1.2)), 1, 1)
            fige.add_trace(go.Scatter(x=bt.equity_usd.index, y=bt.equity_usd,
                                      name=strat_name,
                                      line=dict(color=C.GREEN, width=1.8)), 1, 1)
            fige.add_trace(go.Scatter(x=bt.positions.index, y=bt.positions,
                                      name="Exposure", line=dict(color=C.AMBER, width=0.9,
                                                                 shape="hv"),
                                      fill="tozeroy",
                                      fillcolor="rgba(255,234,0,0.10)"), 2, 1)
            if log_scale:
                fige.update_yaxes(type="log", row=1, col=1)
            fige.update_yaxes(title="Equity ($)", tickprefix="$", row=1, col=1)
            fige.update_yaxes(title="Position", row=2, col=1)
            st.plotly_chart(style_fig(fige, height=520,
                                      title=f"{strat_name} vs buy & hold · {timeframe} bars · "
                                            f"lot ×{lot:g} · {cost_bps:g} bps · "
                                            f"1-bar execution lag"),
                            use_container_width=True)

            u1, u2 = st.columns(2)
            with u1:
                figu = go.Figure()
                figu.add_trace(go.Scatter(x=bt.drawdown.index, y=bt.drawdown, name="Drawdown",
                                          line=dict(color=C.RED, width=1.0),
                                          fill="tozeroy", fillcolor="rgba(255,23,68,0.18)"))
                figu.update_yaxes(tickformat=".0%", title="Drawdown")
                st.plotly_chart(style_fig(figu, height=320, title="Underwater curve"),
                                use_container_width=True)
            with u2:
                rs_win = int(min(max(30, ppy / 2), max(30, len(bt.returns) // 4)))
                rs = qsbt.rolling_sharpe(bt.returns, rs_win, float(rf), ppy)
                figrs = go.Figure()
                figrs.add_trace(go.Scatter(x=rs.index, y=rs, name="Rolling Sharpe",
                                           line=dict(color=C.BLUE, width=1.3)))
                figrs.add_hline(y=0, line=dict(color=C.MUTED, width=1, dash="dot"))
                st.plotly_chart(style_fig(figrs, height=320,
                                          title=f"Rolling Sharpe · {rs_win} bars"),
                                use_container_width=True)

            if timeframe == "1d":
                table = qsbt.monthly_return_table(bt.returns)
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                figm = go.Figure(go.Heatmap(
                    z=table.values, x=month_names, y=table.index.astype(str),
                    colorscale=[[0.0, C.RED], [0.5, C.PANEL_2], [1.0, C.GREEN]], zmid=0.0,
                    texttemplate="%{z:.1%}", textfont=dict(size=10),
                    colorbar=dict(title="Return", tickformat=".0%")))
                figm.update_yaxes(autorange="reversed")
                st.plotly_chart(style_fig(figm, height=90 + 34 * len(table),
                                          title="Monthly returns"),
                                use_container_width=True)

            st.markdown(
                f'<p class="qs-note">Methodology: positions decided at bar t earn bar t+1 '
                f'(no look-ahead) · annualization uses {ppy:,.0f} bars/year for the '
                f'{timeframe} timeframe · Kalman signals use the causal filter only, never '
                f'the smoother · costs charged on turnover, scaled by lot. Filters and '
                f'formulas: ⚙ Settings → Documentation §6. '
                f'Research tool — not investment advice.</p>', unsafe_allow_html=True)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Backtest failed: {exc}")

# ---------------------------------------------------------------------------
# STAGE 8 — Hyperparameter optimization (walk-forward)
# ---------------------------------------------------------------------------

with tab_opt:
    oc1, oc2, oc3, oc4, oc5 = st.columns(5)
    opt_tf = oc1.selectbox("Timeframe", ["1d", "1h", "30m", "15m", "5m"], key="opt_tf")
    opt_strat = oc2.selectbox("Strategy", list(qsopt.OPTIMIZABLE), key="opt_strat")
    opt_cost = oc3.number_input("Cost (bps)", 0.0, 50.0, 5.0, 0.5, key="opt_cost")
    opt_ls = oc4.toggle("Allow shorts", key="opt_ls",
                        disabled=(opt_strat == "Volatility targeting"))
    opt_train = oc5.slider("Train fraction", 0.5, 0.9, 0.7, 0.05, key="opt_train",
                           help="Parameters are ranked on the *test* segment they "
                                "never saw during selection")

    if st.button("🚀 RUN WALK-FORWARD OPTIMIZATION", use_container_width=True,
                 key="opt_run"):
        try:
            close_o = cached_bars(ticker, opt_tf, years)["Close"]
            ppy_o = qsbt.PERIODS_PER_YEAR[opt_tf]
            prog = st.progress(0.0, text="Sweeping parameter grid…")
            df_o = qsopt.grid_search(
                close_o, opt_strat, cost_bps=float(opt_cost), rf=float(rf), ppy=ppy_o,
                long_short=bool(opt_ls) and opt_strat != "Volatility targeting",
                train_frac=float(opt_train),
                progress=lambda f: prog.progress(f, text=f"Sweeping grid… {f:.0%}"))
            prog.empty()
            st.session_state["opt"] = dict(
                df=df_o, strategy=opt_strat, tf=opt_tf, ppy=ppy_o,
                cost=float(opt_cost), ls=bool(opt_ls), train=float(opt_train),
                ticker=ticker, split=df_o.attrs["split_date"],
                n_bars=df_o.attrs["n_bars"], n_combos=df_o.attrs["n_combos"])
        except (ValueError, RuntimeError) as exc:
            st.error(f"Optimization failed: {exc}")

    O = st.session_state.get("opt")
    if O and O["ticker"] == ticker:
        df_o = O["df"]
        best = df_o.iloc[0]
        p1, p2 = qsopt.PARAM_AXES[O["strategy"]]
        best_params = {k: best[k] for k in df_o.columns
                       if k in ("fast", "slow", "ma_type", "resp", "window",
                                "entry_z", "target_vol")}
        label = " · ".join(f"{k}={v:g}" if isinstance(v, (int, float, np.floating))
                           else f"{k}={v}" for k, v in best_params.items())

        om = st.columns(5)
        om[0].metric("Configurations", f"{O['n_combos']:,}")
        om[1].metric("Bar evaluations", f"{O['n_combos'] * O['n_bars']:,}")
        om[2].metric("Best test Sharpe", f"{best['sharpe_test']:.2f}",
                     f"train {best['sharpe_train']:.2f}", delta_color="off")
        om[3].metric("Best test CAGR", pct(best["cagr_test"]))
        om[4].metric("Best test max DD", pct(best["mdd_test"]))
        st.markdown(f'<p class="qs-note">Best configuration — {O["strategy"]}: '
                    f'<b>{label}</b> · ranked on the unseen test segment '
                    f'(after {pd.Timestamp(O["split"]).date()}).</p>',
                    unsafe_allow_html=True)

        pivot = qsopt.surface_pivot(df_o, O["strategy"])
        v1, v2 = st.columns([0.56, 0.44])
        with v1:
            if pivot is not None and pivot.shape[0] > 1 and pivot.shape[1] > 1:
                figs3 = go.Figure(go.Surface(
                    x=pivot.columns.to_numpy(dtype=float),
                    y=pivot.index.to_numpy(dtype=float),
                    z=pivot.to_numpy(), colorscale=BLUE_GREEN_SCALE,
                    colorbar=dict(title="Test Sharpe", len=0.6)))
                figs3.add_trace(go.Scatter3d(
                    x=[float(best[p1])], y=[float(best[p2])],
                    z=[float(best["sharpe_test"])], mode="markers",
                    name="Optimum", marker=dict(size=6, color=C.WHITE,
                                                symbol="diamond")))
                figs3.update_layout(scene=dict(
                    xaxis=dict(title=p1, backgroundcolor=C.BG, gridcolor=C.GRID),
                    yaxis=dict(title=p2, backgroundcolor=C.BG, gridcolor=C.GRID),
                    zaxis=dict(title="Sharpe (test)", backgroundcolor=C.BG,
                               gridcolor=C.GRID),
                    camera=dict(eye=dict(x=1.7, y=-1.7, z=0.8))))
                st.plotly_chart(style_fig(figs3, height=480,
                                          title="Objective landscape · "
                                                "out-of-sample Sharpe"),
                                use_container_width=True)
            else:
                figl = go.Figure()
                xs = df_o.sort_values(p1)
                figl.add_trace(go.Scatter(x=xs[p1], y=xs["sharpe_train"],
                                          name="Train Sharpe",
                                          line=dict(color=C.MUTED, width=1.3)))
                figl.add_trace(go.Scatter(x=xs[p1], y=xs["sharpe_test"],
                                          name="Test Sharpe",
                                          line=dict(color=C.GREEN, width=1.8)))
                figl.add_vline(x=float(best[p1]),
                               line=dict(color=C.WHITE, width=1, dash="dot"))
                figl.update_xaxes(title=p1)
                st.plotly_chart(style_fig(figl, height=480,
                                          title="Objective profile · train vs test"),
                                use_container_width=True)
        with v2:
            figsc = go.Figure()
            lim = float(np.nanmax(np.abs(df_o[["sharpe_train", "sharpe_test"]]
                                         .to_numpy()))) * 1.1 + 0.1
            figsc.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                                       name="No overfit line",
                                       line=dict(color=C.GRID, width=1, dash="dot")))
            figsc.add_trace(go.Scatter(
                x=df_o["sharpe_train"], y=df_o["sharpe_test"], mode="markers",
                name="Configurations",
                marker=dict(size=6, color=df_o["sharpe_test"],
                            colorscale=BLUE_GREEN_SCALE, opacity=0.75)))
            figsc.add_trace(go.Scatter(x=[best["sharpe_train"]],
                                       y=[best["sharpe_test"]], mode="markers",
                                       name="Selected",
                                       marker=dict(symbol="star", size=15,
                                                   color=C.GREEN,
                                                   line=dict(color=C.WHITE, width=1))))
            figsc.update_xaxes(title="Sharpe · train segment")
            figsc.update_yaxes(title="Sharpe · test segment")
            st.plotly_chart(style_fig(figsc, height=480,
                                      title="Overfitting diagnostic — points far "
                                            "below the line memorized the past"),
                            use_container_width=True)

        if pivot is not None and pivot.shape[0] > 1 and pivot.shape[1] > 1:
            fighm = go.Figure(go.Heatmap(
                x=pivot.columns.astype(float), y=pivot.index.astype(float),
                z=pivot.to_numpy(), colorscale=BLUE_GREEN_SCALE,
                colorbar=dict(title="Test Sharpe")))
            fighm.update_xaxes(title=p1)
            fighm.update_yaxes(title=p2)
            st.plotly_chart(style_fig(fighm, height=380,
                                      title="Parameter heatmap · flat bright regions "
                                            "are robust, isolated peaks are luck"),
                            use_container_width=True)

        try:
            close_o = cached_bars(O["ticker"], O["tf"], years)["Close"]
            sig_best = qsopt.build_signal(close_o, O["strategy"], best_params,
                                          O["ls"], O["ppy"])
            res_best = qsbt.run_backtest(close_o, sig_best, cost_bps=O["cost"],
                                         rf=float(rf), periods_per_year=O["ppy"])
            figbe = go.Figure()
            figbe.add_trace(go.Scatter(x=res_best.benchmark_equity.index,
                                       y=res_best.benchmark_equity,
                                       name=f"Buy & hold {O['ticker']}",
                                       line=dict(color=C.MUTED, width=1.2)))
            figbe.add_trace(go.Scatter(x=res_best.equity.index, y=res_best.equity,
                                       name=f"Optimized {O['strategy']}",
                                       line=dict(color=C.GREEN, width=1.8)))
            split_ts = pd.Timestamp(O["split"])
            figbe.add_vrect(x0=split_ts, x1=res_best.equity.index[-1],
                            fillcolor="rgba(0,229,255,0.05)", line_width=0)
            figbe.add_vline(x=split_ts, line=dict(color=C.BLUE, width=1, dash="dot"))
            figbe.add_annotation(x=split_ts, y=1.02, yref="paper",
                                 text="◀ train | test ▶", showarrow=False,
                                 font=dict(color=C.BLUE, size=11))
            figbe.update_yaxes(title="Growth of $1", type="log")
            st.plotly_chart(style_fig(figbe, height=420,
                                      title=f"Best configuration deployed · {label}"),
                            use_container_width=True)
        except (ValueError, RuntimeError) as exc:
            st.info(f"Could not replay best configuration: {exc}")

        show_cols = [c for c in df_o.columns if c not in ("cagr_train",)]
        st.dataframe(df_o[show_cols].head(10).style.format(precision=3),
                     use_container_width=True)
        st.markdown(
            '<p class="qs-note">Methodology: exhaustive causal grid sweep · '
            'chronological split · ranking on the test segment only · both scores '
            'shown so overfitting is visible. A parameter set is trustworthy when '
            'its neighborhood is also good (see heatmap) and train≈test. '
            'Formulas: ⚙ Settings → Documentation §6.</p>', unsafe_allow_html=True)
    elif O:
        st.info("Optimization results were for another ticker — run again.")
    else:
        st.markdown('<p class="qs-note">Pick a strategy and launch the sweep. Every '
                    'configuration is a full causal backtest; the winner is chosen '
                    'on data it never saw.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# STAGE 7 — Probabilistic forecasting
# ---------------------------------------------------------------------------

with tab_fc:
    try:
        close_fc = cached_history(ticker, years)["Close"]
        lr_fc = qsdata.log_returns(close_fc)
        stats_fc = qsdata.compute_stats(close_fc)
    except ValueError as exc:
        st.error(f"Stage 7 failed: {exc}")
        close_fc = None

    if close_fc is not None:
        f1, f2, f3, f4 = st.columns(4)
        horizon = f1.slider("Horizon (trading days)", 21, 504, 126, 21)
        method = f2.selectbox("Method", ["GBM analytic cone", "Block bootstrap (model-free)"])
        drift_src = f3.selectbox("Drift assumption", ["Empirical μ̂", "Risk-free r", "Zero"])
        vol_src = f4.selectbox("Volatility", ["Historical σ̂", "GARCH(1,1) forecast"],
                               disabled=(method != "GBM analytic cone"))

        mu_fc = {"Empirical μ̂": stats_fc.mu_annual, "Risk-free r": float(rf),
                 "Zero": 0.0}[drift_src]
        S0_fc = stats_fc.last_price
        T_fc = horizon / 252.0

        garch = None
        try:
            with st.spinner("Fitting GARCH(1,1) by maximum likelihood…"):
                garch = cached_garch(ticker, years)
        except (ValueError, RuntimeError) as exc:
            st.info(f"GARCH unavailable: {exc}")

        # Integrated volatility per horizon step (flat, or GARCH term structure).
        if vol_src == "GARCH(1,1) forecast" and garch is not None:
            sig_path = qsmodels.garch_forecast_vol(garch, horizon)
            sig_int = np.sqrt(np.cumsum(sig_path**2) / np.arange(1, horizon + 1))
            sigma_T = float(sig_int[-1])
        else:
            sig_int = np.full(horizon, stats_fc.sigma_annual)
            sigma_T = stats_fc.sigma_annual

        t_yrs = np.arange(1, horizon + 1) / 252.0
        rng_fc = np.random.default_rng(33)
        if method == "GBM analytic cone":
            cone = {"t_days": np.arange(1, horizon + 1)}
            for q in qsmodels.QUANTILES:
                z = norm.ppf(q)
                cone[q] = S0_fc * np.exp((mu_fc - 0.5 * sig_int**2) * t_yrs
                                         + sig_int * np.sqrt(t_yrs) * z)
            cone["mean"] = S0_fc * np.exp(mu_fc * t_yrs)
            terminal_fc = S0_fc * np.exp((mu_fc - 0.5 * sigma_T**2) * T_fc
                                         + sigma_T * np.sqrt(T_fc)
                                         * rng_fc.standard_normal(20_000))
        else:
            cone = qsmodels.cone_bootstrap(S0_fc, lr_fc - lr_fc.mean() + mu_fc / 252.0,
                                           horizon, n_sims=10_000)
            terminal_fc = cone["terminal"]

        p_up = float((terminal_fc > S0_fc).mean())
        var_h, es_h = qsmodels.horizon_var_es(terminal_fc, S0_fc, 0.05)
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric(f"P(S > spot) @ {horizon}d", pct(p_up))
        g2.metric("Median forecast", usd(float(cone[0.50][-1])),
                  pct(float(cone[0.50][-1]) / S0_fc - 1))
        g3.metric("VaR 95% (horizon)", pct(var_h))
        g4.metric("ES 95% (horizon)", pct(es_h))
        g5.metric("σ used @ horizon", pct(sigma_T),
                  ("GARCH" if vol_src.startswith("GARCH") and garch else "historical"),
                  delta_color="off")

        future_idx = pd.bdate_range(close_fc.index[-1], periods=horizon + 1)[1:]
        hist_tail = close_fc.iloc[-190:]
        # Attach the cone to the last observed price so forecast and history
        # form one continuous object.
        fx = future_idx.insert(0, close_fc.index[-1])

        def _band(q):
            return np.concatenate(([S0_fc], np.asarray(cone[q])))

        figf = go.Figure()
        figf.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail, name="History",
                                  line=dict(color=C.MUTED, width=1.4)))
        figf.add_trace(go.Scatter(x=fx, y=_band(0.95), showlegend=False,
                                  line=dict(width=0), hoverinfo="skip"))
        figf.add_trace(go.Scatter(x=fx, y=_band(0.05), name="5–95% cone",
                                  fill="tonexty", fillcolor="rgba(0,229,255,0.10)",
                                  line=dict(width=0)))
        figf.add_trace(go.Scatter(x=fx, y=_band(0.75), showlegend=False,
                                  line=dict(width=0), hoverinfo="skip"))
        figf.add_trace(go.Scatter(x=fx, y=_band(0.25), name="25–75% cone",
                                  fill="tonexty", fillcolor="rgba(0,229,255,0.20)",
                                  line=dict(width=0)))
        figf.add_trace(go.Scatter(x=fx, y=_band(0.50), name="Median",
                                  line=dict(color=C.WHITE, width=1.6)))
        figf.add_trace(go.Scatter(x=fx, y=np.concatenate(([S0_fc], cone["mean"])),
                                  name="Mean",
                                  line=dict(color=C.GREEN, width=1.1, dash="dot")))
        figf.add_hline(y=S0_fc, line=dict(color=C.AMBER, width=1, dash="dot"),
                       annotation_text="spot", annotation_font_color=C.AMBER)
        figf.add_vline(x=close_fc.index[-1], line=dict(color=C.MUTED, width=1, dash="dot"))
        # Price targets at the end of the cone.
        for q, color in ((0.95, C.GREEN), (0.50, C.WHITE), (0.05, C.RED)):
            yq = float(cone[q][-1])
            figf.add_annotation(x=future_idx[-1], y=yq, xanchor="left", showarrow=False,
                                text=f" {int(q * 100)}% · {yq:,.2f}",
                                font=dict(color=color, size=11))
        pad = (future_idx[-1] - hist_tail.index[0]) * 0.10
        figf.update_xaxes(range=[hist_tail.index[0], future_idx[-1] + pad])
        st.plotly_chart(style_fig(figf, height=460,
                                  title=f"{ticker} · {horizon}-day probabilistic forecast "
                                        f"({method.lower()}, drift = {drift_src})"),
                        use_container_width=True)

        d1, d2 = st.columns([0.55, 0.45])
        with d1:
            dens = qsmodels.density_surface_gbm(S0_fc, mu_fc, sigma_T, horizon)
            fig3 = go.Figure(go.Surface(
                x=dens["t_years"] * 252, y=dens["s_grid"], z=dens["density"],
                colorscale=BLUE_GREEN_SCALE, opacity=0.95, showscale=False))
            fig3.update_layout(scene=dict(
                xaxis=dict(title="Horizon (days)", backgroundcolor=C.BG, gridcolor=C.GRID),
                yaxis=dict(title="Price", backgroundcolor=C.BG, gridcolor=C.GRID),
                zaxis=dict(title="Density", backgroundcolor=C.BG, gridcolor=C.GRID),
                camera=dict(eye=dict(x=1.7, y=-1.7, z=0.7))))
            st.plotly_chart(style_fig(fig3, height=470,
                                      title="Forward price density p(S, t) — the "
                                            "probability landscape"),
                            use_container_width=True)
        with d2:
            if garch is not None:
                cv = garch["cond_vol_annual"].iloc[-504:]
                fc_vol = qsmodels.garch_forecast_vol(garch, horizon)
                figg = go.Figure()
                figg.add_trace(go.Scatter(x=cv.index, y=cv, name="GARCH conditional σ",
                                          line=dict(color=C.AMBER, width=1.2)))
                figg.add_trace(go.Scatter(x=future_idx, y=fc_vol, name="Forecast E[σ]",
                                          line=dict(color=C.BLUE, width=1.6, dash="dash")))
                figg.add_hline(y=garch["uncond_vol_annual"],
                               line=dict(color=C.MUTED, width=1, dash="dot"),
                               annotation_text="unconditional σ̄",
                               annotation_font_color=C.MUTED)
                figg.update_yaxes(tickformat=".0%", title="Annualized volatility")
                st.plotly_chart(style_fig(figg, height=470,
                                          title=f"GARCH(1,1) · α={garch['alpha']:.3f} "
                                                f"β={garch['beta']:.3f} · persistence "
                                                f"{garch['persistence']:.3f} · half-life "
                                                f"{garch['half_life']:.0f}d"),
                                use_container_width=True)
            else:
                st.info("GARCH fit unavailable for this series.")
        st.markdown(
            '<p class="qs-note">The cone shows quantiles of the forecast distribution, '
            'not a point prediction — the honest way to state where a price can go. '
            'Bootstrap resamples 5-day blocks of real returns (fat tails preserved); '
            'GARCH volatility mean-reverts to σ̄ at its estimated half-life. '
            'Formulas: ⚙ Settings → Documentation §7–8. Not investment advice.</p>',
            unsafe_allow_html=True)
