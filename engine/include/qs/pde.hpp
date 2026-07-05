#pragma once
// QuantSphere Engine — Black-Scholes PDE solver.
// Crank-Nicolson on a uniform S-grid; Thomas O(N) tridiagonal solve for
// European exercise, PSOR (projected SOR) for the American LCP.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace qs {

struct PdeResult {
    double price = 0.0;
    double delta = 0.0;
    double gamma = 0.0;
    double theta = 0.0;            // per year, at (S0, t=0)
    std::vector<double> s_grid;    // Ns+1
    std::vector<double> t_grid;    // Nt+1 (calendar time, 0 -> T)
    std::vector<double> surface;   // row-major (Nt+1) x (Ns+1), V(t_j, S_i)
};

namespace detail {

// Thomas algorithm for A x = d with A tridiagonal (a: sub, b: diag, c: super).
inline void thomas(const std::vector<double>& a, const std::vector<double>& b,
                   const std::vector<double>& c, std::vector<double>& d,
                   std::vector<double>& x, std::vector<double>& cp, std::vector<double>& dp) {
    const std::size_t n = b.size();
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for (std::size_t i = 1; i < n; ++i) {
        const double m = b[i] - a[i] * cp[i - 1];
        cp[i] = c[i] / m;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }
    x[n - 1] = dp[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) x[i] = dp[i] - cp[i] * x[i + 1];
}

// PSOR for the American linear complementarity problem: A x >= d, x >= psi,
// (A x - d)'(x - psi) = 0, with A the Crank-Nicolson left-hand matrix.
inline void psor(const std::vector<double>& a, const std::vector<double>& b,
                 const std::vector<double>& c, const std::vector<double>& d,
                 const std::vector<double>& psi, std::vector<double>& x) {
    const std::size_t n = b.size();
    const double omega = 1.2, tol = 1e-9;
    for (int it = 0; it < 4000; ++it) {
        double err = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double resid = d[i] - b[i] * x[i];
            if (i > 0) resid -= a[i] * x[i - 1];
            if (i + 1 < n) resid -= c[i] * x[i + 1];
            const double xn = std::max(psi[i], x[i] + omega * resid / b[i]);
            err = std::max(err, std::fabs(xn - x[i]));
            x[i] = xn;
        }
        if (err < tol) break;
    }
}

} // namespace detail

inline PdeResult pde_bs(double S0, double K, double r, double sigma, double T, bool is_call,
                        bool american, int Ns = 200, int Nt = 200, double s_max_mult = 3.0) {
    if (S0 <= 0.0 || K <= 0.0 || T <= 0.0) throw std::invalid_argument("S0, K, T must be positive");
    if (sigma <= 0.0) throw std::invalid_argument("sigma must be positive");
    if (Ns < 10 || Nt < 10) throw std::invalid_argument("Ns and Nt must be >= 10");

    // Ensure the truncation boundary comfortably contains the diffusion cone.
    const double smax =
        std::max(s_max_mult, std::exp(3.0 * sigma * std::sqrt(T))) * std::max(S0, K);
    const double dS = smax / Ns;
    const double dt = T / Nt;

    PdeResult res;
    res.s_grid.resize(Ns + 1);
    res.t_grid.resize(Nt + 1);
    for (int i = 0; i <= Ns; ++i) res.s_grid[i] = i * dS;
    for (int j = 0; j <= Nt; ++j) res.t_grid[j] = j * dt;
    res.surface.assign(std::size_t(Nt + 1) * (Ns + 1), 0.0);

    auto payoff = [&](double S) { return is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0); };

    // Terminal layer (t = T).
    std::vector<double> V(Ns + 1);
    for (int i = 0; i <= Ns; ++i) V[i] = payoff(res.s_grid[i]);
    std::copy(V.begin(), V.end(), res.surface.begin() + std::size_t(Nt) * (Ns + 1));

    // Crank-Nicolson coefficients for interior nodes i = 1..Ns-1.
    const int n = Ns - 1;
    std::vector<double> la(n), lb(n), lc(n); // left  (I - M/2)
    std::vector<double> ra(n), rb(n), rc(n); // right (I + M/2)
    for (int k = 0; k < n; ++k) {
        const double i = k + 1.0;
        const double A = 0.25 * dt * (sigma * sigma * i * i - r * i);
        const double B = -0.5 * dt * (sigma * sigma * i * i + r);
        const double C = 0.25 * dt * (sigma * sigma * i * i + r * i);
        la[k] = -A; lb[k] = 1.0 - B; lc[k] = -C;
        ra[k] = A;  rb[k] = 1.0 + B; rc[k] = C;
    }

    std::vector<double> rhs(n), x(n), psi(n), cp(n), dp(n);
    for (int k = 0; k < n; ++k) psi[k] = payoff(res.s_grid[k + 1]);

    auto lower_bc = [&](double tau) {
        if (is_call) return 0.0;
        return american ? K : K * std::exp(-r * tau);
    };
    auto upper_bc = [&](double tau) {
        if (!is_call) return 0.0;
        return smax - K * std::exp(-r * tau);
    };

    // March backward in calendar time: layer j holds V(t = j*dt, .).
    for (int j = Nt - 1; j >= 0; --j) {
        const double tau_new = T - j * dt;        // time-to-maturity of the layer being computed
        const double tau_old = T - (j + 1) * dt;  // layer we are stepping from

        for (int k = 0; k < n; ++k) {
            rhs[k] = ra[k] * V[k] + rb[k] * V[k + 1] + rc[k] * V[k + 2];
        }
        // Boundary contributions from both time layers (Crank-Nicolson averaging).
        rhs[0]     += ra[0] * lower_bc(tau_old) - la[0] * lower_bc(tau_new);
        rhs[n - 1] += rc[n - 1] * upper_bc(tau_old) - lc[n - 1] * upper_bc(tau_new);

        if (american) {
            for (int k = 0; k < n; ++k) x[k] = std::max(V[k + 1], psi[k]);
            detail::psor(la, lb, lc, rhs, psi, x);
        } else {
            detail::thomas(la, lb, lc, rhs, x, cp, dp);
        }

        V[0] = lower_bc(tau_new);
        V[Ns] = upper_bc(tau_new);
        for (int k = 0; k < n; ++k) V[k + 1] = x[k];
        std::copy(V.begin(), V.end(), res.surface.begin() + std::size_t(j) * (Ns + 1));
    }

    // Interpolate price and Greeks at S0 from the t=0 layer.
    auto value_at = [&](const double* layer, double S) {
        const double pos = std::clamp(S / dS, 0.0, double(Ns));
        const int i = std::min(int(pos), Ns - 1);
        const double w = pos - i;
        return layer[i] * (1.0 - w) + layer[i + 1] * w;
    };
    const double* L0 = res.surface.data();
    const double* L1 = res.surface.data() + (Ns + 1);
    res.price = value_at(L0, S0);
    const double h = dS;
    const double vu = value_at(L0, std::min(S0 + h, smax));
    const double vd = value_at(L0, std::max(S0 - h, 0.0));
    res.delta = (vu - vd) / (2.0 * h);
    res.gamma = (vu - 2.0 * res.price + vd) / (h * h);
    res.theta = (value_at(L1, S0) - res.price) / dt;
    return res;
}

} // namespace qs
