#pragma once
// QuantSphere Engine — multithreaded Monte Carlo under GBM, Heston,
// Merton jump-diffusion and Variance-Gamma dynamics (risk-neutral measure).

#include "rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace qs {

enum class Model { GBM, Heston, Merton, VG };

enum class BarrierType { None, UpOut, DownOut, UpIn, DownIn };

struct McParams {
    double S0 = 100.0;
    double r = 0.02;
    double T = 1.0;
    double sigma = 0.2; // GBM / Merton diffusive vol, VG gaussian vol
    // Heston
    double v0 = 0.04, kappa = 1.5, theta = 0.04, xi = 0.5, rho = -0.7;
    // Merton jumps
    double lam = 0.5, mu_j = -0.05, sig_j = 0.15;
    // Variance Gamma
    double vg_theta = -0.1, vg_nu = 0.2;
};

inline Model parse_model(const std::string& name) {
    if (name == "gbm") return Model::GBM;
    if (name == "heston") return Model::Heston;
    if (name == "merton") return Model::Merton;
    if (name == "vg") return Model::VG;
    throw std::invalid_argument("unknown model: " + name);
}

inline BarrierType parse_barrier(const std::string& name) {
    if (name == "none") return BarrierType::None;
    if (name == "up-and-out") return BarrierType::UpOut;
    if (name == "down-and-out") return BarrierType::DownOut;
    if (name == "up-and-in") return BarrierType::UpIn;
    if (name == "down-and-in") return BarrierType::DownIn;
    throw std::invalid_argument("unknown barrier type: " + name);
}

inline void validate(const McParams& p, Model m) {
    if (p.S0 <= 0.0 || p.T <= 0.0) throw std::invalid_argument("S0 and T must be positive");
    if (m == Model::GBM || m == Model::Merton || m == Model::VG) {
        if (p.sigma <= 0.0) throw std::invalid_argument("sigma must be positive");
    }
    if (m == Model::Heston) {
        if (p.v0 < 0.0 || p.theta < 0.0 || p.kappa < 0.0 || p.xi < 0.0)
            throw std::invalid_argument("Heston parameters must be non-negative");
        if (p.rho < -1.0 || p.rho > 1.0) throw std::invalid_argument("rho must lie in [-1,1]");
    }
    if (m == Model::VG) {
        if (p.vg_nu <= 0.0) throw std::invalid_argument("vg_nu must be positive");
        if (1.0 - p.vg_theta * p.vg_nu - 0.5 * p.sigma * p.sigma * p.vg_nu <= 0.0)
            throw std::invalid_argument(
                "VG martingale correction undefined: require 1 - theta*nu - 0.5*sigma^2*nu > 0");
    }
}

// Simulates one path of log-price into `logs` (steps+1 entries, logs[0] = ln S0).
// Heston carries its variance state internally; VG uses a gamma time change.
inline void simulate_one(Model m, const McParams& p, int steps, Rng& rng, double* logs) {
    const double dt = p.T / steps;
    const double sqdt = std::sqrt(dt);
    double x = std::log(p.S0);
    logs[0] = x;

    switch (m) {
    case Model::GBM: {
        const double drift = (p.r - 0.5 * p.sigma * p.sigma) * dt;
        const double vol = p.sigma * sqdt;
        for (int i = 1; i <= steps; ++i) {
            x += drift + vol * rng.normal();
            logs[i] = x;
        }
        break;
    }
    case Model::Heston: {
        // Full-truncation Euler: variance may go negative in the state but
        // only its positive part enters drift and diffusion.
        double v = p.v0;
        const double srho = std::sqrt(1.0 - p.rho * p.rho);
        for (int i = 1; i <= steps; ++i) {
            const double vp = std::max(v, 0.0);
            const double z1 = rng.normal();
            const double z2 = p.rho * z1 + srho * rng.normal();
            x += (p.r - 0.5 * vp) * dt + std::sqrt(vp) * sqdt * z1;
            v += p.kappa * (p.theta - vp) * dt + p.xi * std::sqrt(vp) * sqdt * z2;
            logs[i] = x;
        }
        break;
    }
    case Model::Merton: {
        // Compensated jumps keep the discounted price a martingale.
        const double k = std::exp(p.mu_j + 0.5 * p.sig_j * p.sig_j) - 1.0;
        const double drift = (p.r - 0.5 * p.sigma * p.sigma - p.lam * k) * dt;
        const double vol = p.sigma * sqdt;
        for (int i = 1; i <= steps; ++i) {
            double jump = 0.0;
            const int n = rng.poisson(p.lam * dt);
            if (n > 0) jump = n * p.mu_j + p.sig_j * std::sqrt(double(n)) * rng.normal();
            x += drift + vol * rng.normal() + jump;
            logs[i] = x;
        }
        break;
    }
    case Model::VG: {
        // X_t = theta*G + sigma*W(G) with G a gamma subordinator;
        // omega is the martingale (exponential compensator) correction.
        const double omega =
            std::log(1.0 - p.vg_theta * p.vg_nu - 0.5 * p.sigma * p.sigma * p.vg_nu) / p.vg_nu;
        const double drift = (p.r + omega) * dt;
        for (int i = 1; i <= steps; ++i) {
            const double g = rng.gamma(dt / p.vg_nu, p.vg_nu);
            x += drift + p.vg_theta * g + p.sigma * std::sqrt(g) * rng.normal();
            logs[i] = x;
        }
        break;
    }
    }
}

namespace detail {

template <typename Fn>
inline void parallel_units(std::uint64_t n_paths, Fn&& fn) {
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned n_threads = std::min<unsigned>(hw, 16u);
    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) {
        pool.emplace_back([&, t] {
            for (int u = int(t); u < WORK_UNITS; u += int(n_threads)) {
                const std::uint64_t lo = n_paths * u / WORK_UNITS;
                const std::uint64_t hi = n_paths * (u + 1) / WORK_UNITS;
                if (hi > lo) fn(u, lo, hi);
            }
        });
    }
    for (auto& th : pool) th.join();
}

} // namespace detail

// Full path matrix (row-major, n_paths x (steps+1)) in *price* space.
// Intended for visualization: keep n_paths modest (<= a few thousand).
inline std::vector<double> simulate_paths(Model m, const McParams& p, std::uint64_t n_paths,
                                          int steps, std::uint64_t seed) {
    validate(p, m);
    if (steps < 1 || n_paths < 1) throw std::invalid_argument("steps and n_paths must be >= 1");
    std::vector<double> out(n_paths * (steps + 1));
    detail::parallel_units(n_paths, [&](int unit, std::uint64_t lo, std::uint64_t hi) {
        Rng rng(seed, std::uint64_t(unit));
        std::vector<double> logs(steps + 1);
        for (std::uint64_t j = lo; j < hi; ++j) {
            simulate_one(m, p, steps, rng, logs.data());
            double* row = out.data() + j * (steps + 1);
            for (int i = 0; i <= steps; ++i) row[i] = std::exp(logs[i]);
        }
    });
    return out;
}

struct McPriceResult {
    double price = 0.0;
    double std_error = 0.0;
    std::vector<double> terminal; // terminal prices, for histograms / VaR
};

// European (and optionally barrier-conditioned) pricing by Monte Carlo.
// Payoffs use discrete barrier monitoring on the simulation grid.
inline McPriceResult mc_price(Model m, const McParams& p, double K, bool is_call,
                              std::uint64_t n_paths, int steps, std::uint64_t seed,
                              BarrierType bt = BarrierType::None, double barrier = 0.0) {
    validate(p, m);
    if (K <= 0.0) throw std::invalid_argument("strike must be positive");
    if (steps < 1 || n_paths < 1) throw std::invalid_argument("steps and n_paths must be >= 1");
    if (bt != BarrierType::None && barrier <= 0.0)
        throw std::invalid_argument("barrier level must be positive");

    McPriceResult res;
    res.terminal.resize(n_paths);
    std::vector<double> sum(WORK_UNITS, 0.0), sum2(WORK_UNITS, 0.0);
    const double log_barrier = (bt != BarrierType::None) ? std::log(barrier) : 0.0;

    detail::parallel_units(n_paths, [&](int unit, std::uint64_t lo, std::uint64_t hi) {
        Rng rng(seed, std::uint64_t(unit));
        std::vector<double> logs(steps + 1);
        double s = 0.0, s2 = 0.0;
        for (std::uint64_t j = lo; j < hi; ++j) {
            simulate_one(m, p, steps, rng, logs.data());
            const double ST = std::exp(logs[steps]);
            res.terminal[j] = ST;

            bool alive = true;
            if (bt != BarrierType::None) {
                bool touched = false;
                if (bt == BarrierType::UpOut || bt == BarrierType::UpIn) {
                    for (int i = 0; i <= steps; ++i)
                        if (logs[i] >= log_barrier) { touched = true; break; }
                } else {
                    for (int i = 0; i <= steps; ++i)
                        if (logs[i] <= log_barrier) { touched = true; break; }
                }
                const bool knock_in = (bt == BarrierType::UpIn || bt == BarrierType::DownIn);
                alive = knock_in ? touched : !touched;
            }

            double payoff = 0.0;
            if (alive) payoff = is_call ? std::max(ST - K, 0.0) : std::max(K - ST, 0.0);
            s += payoff;
            s2 += payoff * payoff;
        }
        sum[unit] += s;
        sum2[unit] += s2;
    });

    double s = 0.0, s2 = 0.0;
    for (int u = 0; u < WORK_UNITS; ++u) { s += sum[u]; s2 += sum2[u]; }
    const double n = double(n_paths);
    const double disc = std::exp(-p.r * p.T);
    const double mean = s / n;
    const double var = std::max(0.0, s2 / n - mean * mean);
    res.price = disc * mean;
    res.std_error = disc * std::sqrt(var / n);
    return res;
}

} // namespace qs
