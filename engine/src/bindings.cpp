// QuantSphere Engine — pybind11 bindings.
// All heavy loops release the GIL; arrays cross the boundary as NumPy buffers.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "qs/iv.hpp"
#include "qs/kalman.hpp"
#include "qs/mc.hpp"
#include "qs/pde.hpp"

#include <thread>

namespace py = pybind11;

namespace {

qs::McParams make_params(double S0, double r, double T, double sigma, double v0, double kappa,
                         double theta, double xi, double rho, double lam, double mu_j,
                         double sig_j, double vg_theta, double vg_nu) {
    qs::McParams p;
    p.S0 = S0; p.r = r; p.T = T; p.sigma = sigma;
    p.v0 = v0; p.kappa = kappa; p.theta = theta; p.xi = xi; p.rho = rho;
    p.lam = lam; p.mu_j = mu_j; p.sig_j = sig_j;
    p.vg_theta = vg_theta; p.vg_nu = vg_nu;
    return p;
}

py::array_t<double> to_array_2d(std::vector<double>&& v, std::size_t rows, std::size_t cols) {
    auto* holder = new std::vector<double>(std::move(v));
    py::capsule free_when_done(holder, [](void* p) { delete static_cast<std::vector<double>*>(p); });
    return py::array_t<double>({rows, cols},
                               {cols * sizeof(double), sizeof(double)},
                               holder->data(), free_when_done);
}

py::array_t<double> to_array_1d(std::vector<double>&& v) {
    auto* holder = new std::vector<double>(std::move(v));
    py::capsule free_when_done(holder, [](void* p) { delete static_cast<std::vector<double>*>(p); });
    return py::array_t<double>({holder->size()}, {sizeof(double)}, holder->data(), free_when_done);
}

} // namespace

PYBIND11_MODULE(qsengine, m) {
    m.doc() = "QuantSphere native engine: Monte Carlo, PDE, Kalman, implied volatility";

    m.def("version", [] { return std::string("1.0.0"); });
    m.def("hardware_threads", [] { return std::thread::hardware_concurrency(); });

    m.def(
        "simulate_paths",
        [](const std::string& model, std::uint64_t n_paths, int steps, std::uint64_t seed,
           double S0, double r, double T, double sigma, double v0, double kappa, double theta,
           double xi, double rho, double lam, double mu_j, double sig_j, double vg_theta,
           double vg_nu) {
            const qs::Model mo = qs::parse_model(model);
            const qs::McParams p = make_params(S0, r, T, sigma, v0, kappa, theta, xi, rho, lam,
                                               mu_j, sig_j, vg_theta, vg_nu);
            std::vector<double> paths;
            {
                py::gil_scoped_release release;
                paths = qs::simulate_paths(mo, p, n_paths, steps, seed);
            }
            return to_array_2d(std::move(paths), n_paths, std::size_t(steps) + 1);
        },
        py::arg("model"), py::arg("n_paths"), py::arg("steps"), py::arg("seed") = 42,
        py::arg("S0") = 100.0, py::arg("r") = 0.02, py::arg("T") = 1.0, py::arg("sigma") = 0.2,
        py::arg("v0") = 0.04, py::arg("kappa") = 1.5, py::arg("theta") = 0.04,
        py::arg("xi") = 0.5, py::arg("rho") = -0.7, py::arg("lam") = 0.5,
        py::arg("mu_j") = -0.05, py::arg("sig_j") = 0.15, py::arg("vg_theta") = -0.1,
        py::arg("vg_nu") = 0.2,
        "Full price-path matrix (n_paths x steps+1), risk-neutral dynamics");

    m.def(
        "mc_price",
        [](const std::string& model, double K, bool is_call, std::uint64_t n_paths, int steps,
           std::uint64_t seed, const std::string& barrier_type, double barrier, double S0,
           double r, double T, double sigma, double v0, double kappa, double theta, double xi,
           double rho, double lam, double mu_j, double sig_j, double vg_theta, double vg_nu) {
            const qs::Model mo = qs::parse_model(model);
            const qs::BarrierType bt = qs::parse_barrier(barrier_type);
            const qs::McParams p = make_params(S0, r, T, sigma, v0, kappa, theta, xi, rho, lam,
                                               mu_j, sig_j, vg_theta, vg_nu);
            qs::McPriceResult res;
            {
                py::gil_scoped_release release;
                res = qs::mc_price(mo, p, K, is_call, n_paths, steps, seed, bt, barrier);
            }
            py::dict d;
            d["price"] = res.price;
            d["std_error"] = res.std_error;
            d["terminal"] = to_array_1d(std::move(res.terminal));
            return d;
        },
        py::arg("model"), py::arg("K"), py::arg("is_call"), py::arg("n_paths") = 50000,
        py::arg("steps") = 252, py::arg("seed") = 42, py::arg("barrier_type") = "none",
        py::arg("barrier") = 0.0, py::arg("S0") = 100.0, py::arg("r") = 0.02,
        py::arg("T") = 1.0, py::arg("sigma") = 0.2, py::arg("v0") = 0.04,
        py::arg("kappa") = 1.5, py::arg("theta") = 0.04, py::arg("xi") = 0.5,
        py::arg("rho") = -0.7, py::arg("lam") = 0.5, py::arg("mu_j") = -0.05,
        py::arg("sig_j") = 0.15, py::arg("vg_theta") = -0.1, py::arg("vg_nu") = 0.2,
        "Monte Carlo option price with standard error and terminal distribution");

    m.def(
        "pde_price",
        [](double S0, double K, double r, double sigma, double T, bool is_call, bool american,
           int Ns, int Nt, double s_max_mult) {
            qs::PdeResult res;
            {
                py::gil_scoped_release release;
                res = qs::pde_bs(S0, K, r, sigma, T, is_call, american, Ns, Nt, s_max_mult);
            }
            const std::size_t rows = res.t_grid.size(), cols = res.s_grid.size();
            py::dict d;
            d["price"] = res.price;
            d["delta"] = res.delta;
            d["gamma"] = res.gamma;
            d["theta"] = res.theta;
            d["s_grid"] = to_array_1d(std::move(res.s_grid));
            d["t_grid"] = to_array_1d(std::move(res.t_grid));
            d["surface"] = to_array_2d(std::move(res.surface), rows, cols);
            return d;
        },
        py::arg("S0"), py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("T"),
        py::arg("is_call") = true, py::arg("american") = false, py::arg("Ns") = 200,
        py::arg("Nt") = 200, py::arg("s_max_mult") = 3.0,
        "Crank-Nicolson Black-Scholes PDE solve (PSOR for American exercise)");

    m.def(
        "kalman",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> y,
           const std::string& model, double q, double r, double q_drift) {
            auto buf = y.request();
            if (buf.ndim != 1) throw std::invalid_argument("y must be 1-D");
            std::vector<double> yv(static_cast<double*>(buf.ptr),
                                   static_cast<double*>(buf.ptr) + buf.shape[0]);
            qs::KalmanSpec spec;
            if (model == "local_level") spec = qs::local_level(yv.empty() ? 0.0 : yv[0], q, r);
            else if (model == "local_trend")
                spec = qs::local_trend(yv.empty() ? 0.0 : yv[0], q, q_drift, r);
            else throw std::invalid_argument("model must be 'local_level' or 'local_trend'");

            qs::KalmanResult res;
            {
                py::gil_scoped_release release;
                res = qs::kalman_run(spec, yv);
            }
            const std::size_t n = yv.size(), d = std::size_t(spec.dim);
            py::dict out;
            out["x_filt"] = to_array_2d(std::move(res.x_filt), n, d);
            out["P_filt"] = to_array_2d(std::move(res.P_filt), n, d * d);
            out["x_smooth"] = to_array_2d(std::move(res.x_smooth), n, d);
            out["P_smooth"] = to_array_2d(std::move(res.P_smooth), n, d * d);
            out["innovations"] = to_array_1d(std::move(res.innovations));
            out["innov_var"] = to_array_1d(std::move(res.innov_var));
            out["loglik"] = res.loglik;
            out["dim"] = spec.dim;
            return out;
        },
        py::arg("y"), py::arg("model") = "local_level", py::arg("q") = 1e-5,
        py::arg("r") = 1e-3, py::arg("q_drift") = 1e-8,
        "Kalman filter + RTS smoother on a scalar series");

    m.def(
        "bs_price",
        [](double S, double K, double T, double r, double sigma, bool is_call) {
            return qs::bs_price(S, K, T, r, sigma, is_call);
        },
        py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
        py::arg("is_call") = true);

    m.def(
        "implied_vol",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> prices,
           double S,
           py::array_t<double, py::array::c_style | py::array::forcecast> strikes,
           py::array_t<double, py::array::c_style | py::array::forcecast> maturities,
           double r,
           py::array_t<bool, py::array::c_style | py::array::forcecast> is_call) {
            auto pb = prices.request(), kb = strikes.request(), tb = maturities.request(),
                 cb = is_call.request();
            const auto n = pb.shape[0];
            if (kb.shape[0] != n || tb.shape[0] != n || cb.shape[0] != n)
                throw std::invalid_argument("all arrays must share the same length");
            std::vector<double> out(n);
            {
                py::gil_scoped_release release;
                const double* pp = static_cast<double*>(pb.ptr);
                const double* kk = static_cast<double*>(kb.ptr);
                const double* tt = static_cast<double*>(tb.ptr);
                const bool* cc = static_cast<bool*>(cb.ptr);
                for (py::ssize_t i = 0; i < n; ++i)
                    out[i] = qs::implied_vol(pp[i], S, kk[i], tt[i], r, cc[i]);
            }
            return to_array_1d(std::move(out));
        },
        py::arg("prices"), py::arg("S"), py::arg("strikes"), py::arg("maturities"),
        py::arg("r"), py::arg("is_call"),
        "Vectorized implied volatility (Newton + bisection fallback); NaN when no root");
}
