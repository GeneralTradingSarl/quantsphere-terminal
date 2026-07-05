#pragma once
// QuantSphere Engine — Black-Scholes analytics and implied volatility.
// Newton-Raphson with vega, hardened by bisection fallback on [1e-6, 5].

#include <cmath>
#include <limits>

namespace qs {

inline double norm_cdf(double x) { return 0.5 * std::erfc(-x * 0.7071067811865475244); }

inline double norm_pdf(double x) {
    return 0.3989422804014326779 * std::exp(-0.5 * x * x);
}

inline double bs_price(double S, double K, double T, double r, double sigma, bool is_call) {
    if (S <= 0.0 || K <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    if (T <= 0.0 || sigma <= 0.0) {
        const double fwd_payoff = is_call ? S - K * std::exp(-r * T) : K * std::exp(-r * T) - S;
        return fwd_payoff > 0.0 ? fwd_payoff : 0.0;
    }
    const double sq = sigma * std::sqrt(T);
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq;
    const double d2 = d1 - sq;
    if (is_call) return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

inline double bs_vega(double S, double K, double T, double r, double sigma) {
    if (S <= 0.0 || K <= 0.0 || T <= 0.0 || sigma <= 0.0) return 0.0;
    const double sq = sigma * std::sqrt(T);
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq;
    return S * norm_pdf(d1) * std::sqrt(T);
}

inline double implied_vol(double price, double S, double K, double T, double r, bool is_call) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(price > 0.0) || S <= 0.0 || K <= 0.0 || T <= 0.0) return nan;

    const double lo_v = 1e-6, hi_v = 5.0;
    const double lower = bs_price(S, K, T, r, lo_v, is_call);
    const double upper = bs_price(S, K, T, r, hi_v, is_call);
    if (price <= lower + 1e-12 || price >= upper - 1e-12) return nan; // outside no-arbitrage band

    // Newton from a Brenner-Subrahmanyam style start.
    double sigma = std::sqrt(2.0 * 3.14159265358979323846 / T) * price / S;
    sigma = std::fmin(std::fmax(sigma, 0.05), 2.0);
    for (int it = 0; it < 50; ++it) {
        const double diff = bs_price(S, K, T, r, sigma, is_call) - price;
        if (std::fabs(diff) < 1e-10) return sigma;
        const double vega = bs_vega(S, K, T, r, sigma);
        if (vega < 1e-12) break;
        const double step = diff / vega;
        sigma -= step;
        if (sigma <= lo_v || sigma >= hi_v || !std::isfinite(sigma)) break;
        if (std::fabs(step) < 1e-12) return sigma;
    }

    // Bisection fallback (price is monotone in sigma).
    double a = lo_v, b = hi_v;
    for (int it = 0; it < 200; ++it) {
        const double m = 0.5 * (a + b);
        const double pm = bs_price(S, K, T, r, m, is_call);
        if (std::fabs(pm - price) < 1e-10) return m;
        if (pm < price) a = m; else b = m;
    }
    return 0.5 * (a + b);
}

} // namespace qs
