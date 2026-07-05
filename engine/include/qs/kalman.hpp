#pragma once
// QuantSphere Engine — linear-Gaussian Kalman filter + RTS smoother for
// scalar observations and small state dimension (1..4). Row-major matrices.

#include <cmath>
#include <stdexcept>
#include <vector>

namespace qs {

struct KalmanSpec {
    int dim = 1;
    std::vector<double> A;  // dim x dim transition
    std::vector<double> C;  // 1 x dim observation
    std::vector<double> Q;  // dim x dim process noise
    double R = 1.0;         // observation noise variance
    std::vector<double> x0; // dim initial state
    std::vector<double> P0; // dim x dim initial covariance
};

struct KalmanResult {
    // n x dim filtered/smoothed means; n x dim*dim covariances.
    std::vector<double> x_filt, P_filt, x_smooth, P_smooth;
    std::vector<double> innovations, innov_var; // n each
    double loglik = 0.0;
};

namespace detail {

inline void matmul(const double* A, const double* B, double* out, int n, int m, int p) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j) {
            double s = 0.0;
            for (int k = 0; k < m; ++k) s += A[i * m + k] * B[k * p + j];
            out[i * p + j] = s;
        }
}

inline void transpose(const double* A, double* out, int n, int m) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) out[j * n + i] = A[i * m + j];
}

// Solve X * A = B for X (m x d), i.e. X = B * A^{-1}. Equivalent to
// A^T X^T = B^T; Gauss-Jordan with partial pivoting on A^T, m RHS columns.
// Only used for d <= 4; A is a covariance with diagonal jitter.
inline void solve_right(const double* A, const double* B, double* X, int d, int m) {
    std::vector<double> M(d * d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) M[i * d + j] = A[j * d + i];
    std::vector<double> R(d * m);
    for (int r2 = 0; r2 < m; ++r2)
        for (int i = 0; i < d; ++i) R[i * m + r2] = B[r2 * d + i];

    for (int col = 0; col < d; ++col) {
        int piv = col;
        for (int r3 = col + 1; r3 < d; ++r3)
            if (std::fabs(M[r3 * d + col]) > std::fabs(M[piv * d + col])) piv = r3;
        if (std::fabs(M[piv * d + col]) < 1e-300) throw std::runtime_error("singular matrix in RTS");
        if (piv != col) {
            for (int j = 0; j < d; ++j) std::swap(M[col * d + j], M[piv * d + j]);
            for (int j = 0; j < m; ++j) std::swap(R[col * m + j], R[piv * m + j]);
        }
        const double p = M[col * d + col];
        for (int j = 0; j < d; ++j) M[col * d + j] /= p;
        for (int j = 0; j < m; ++j) R[col * m + j] /= p;
        for (int r3 = 0; r3 < d; ++r3) {
            if (r3 == col) continue;
            const double f = M[r3 * d + col];
            if (f == 0.0) continue;
            for (int j = 0; j < d; ++j) M[r3 * d + j] -= f * M[col * d + j];
            for (int j = 0; j < m; ++j) R[r3 * m + j] -= f * R[col * m + j];
        }
    }
    for (int r2 = 0; r2 < m; ++r2)
        for (int i = 0; i < d; ++i) X[r2 * d + i] = R[i * m + r2];
}

} // namespace detail

inline KalmanResult kalman_run(const KalmanSpec& spec, const std::vector<double>& y) {
    const int d = spec.dim;
    const int n = int(y.size());
    if (d < 1 || d > 4) throw std::invalid_argument("state dimension must be in 1..4");
    if (n < 2) throw std::invalid_argument("need at least 2 observations");
    if (int(spec.A.size()) != d * d || int(spec.Q.size()) != d * d ||
        int(spec.C.size()) != d || int(spec.x0.size()) != d || int(spec.P0.size()) != d * d)
        throw std::invalid_argument("inconsistent Kalman spec dimensions");
    if (spec.R <= 0.0) throw std::invalid_argument("R must be positive");

    KalmanResult res;
    res.x_filt.assign(std::size_t(n) * d, 0.0);
    res.P_filt.assign(std::size_t(n) * d * d, 0.0);
    res.innovations.assign(n, 0.0);
    res.innov_var.assign(n, 0.0);

    std::vector<double> x_pred_all(std::size_t(n) * d), P_pred_all(std::size_t(n) * d * d);
    std::vector<double> x(spec.x0), P(spec.P0);
    std::vector<double> xp(d), Pp(d * d), At(d * d), tmp(d * d), K(d);

    detail::transpose(spec.A.data(), At.data(), d, d);
    const double LOG2PI = 1.8378770664093454836;

    for (int t = 0; t < n; ++t) {
        // Predict.
        if (t == 0) {
            xp = x; Pp = P; // prior enters the first update directly
        } else {
            detail::matmul(spec.A.data(), x.data(), xp.data(), d, d, 1);
            detail::matmul(spec.A.data(), P.data(), tmp.data(), d, d, d);
            detail::matmul(tmp.data(), At.data(), Pp.data(), d, d, d);
            for (int i = 0; i < d * d; ++i) Pp[i] += spec.Q[i];
        }
        std::copy(xp.begin(), xp.end(), x_pred_all.begin() + std::size_t(t) * d);
        std::copy(Pp.begin(), Pp.end(), P_pred_all.begin() + std::size_t(t) * d * d);

        // Update with scalar observation.
        double yhat = 0.0;
        for (int i = 0; i < d; ++i) yhat += spec.C[i] * xp[i];
        double S = spec.R;
        std::vector<double> PCt(d, 0.0);
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) PCt[i] += Pp[i * d + j] * spec.C[j];
            S += spec.C[i] * PCt[i];
        }
        S = std::max(S, 1e-300);
        const double v = y[t] - yhat;
        for (int i = 0; i < d; ++i) K[i] = PCt[i] / S;

        for (int i = 0; i < d; ++i) x[i] = xp[i] + K[i] * v;
        // Joseph-free form is fine here: P = (I - K C) Pp, then symmetrize.
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) P[i * d + j] = Pp[i * d + j] - K[i] * PCt[j];
        for (int i = 0; i < d; ++i)
            for (int j = i + 1; j < d; ++j) {
                const double m = 0.5 * (P[i * d + j] + P[j * d + i]);
                P[i * d + j] = P[j * d + i] = m;
            }

        res.innovations[t] = v;
        res.innov_var[t] = S;
        res.loglik += -0.5 * (LOG2PI + std::log(S) + v * v / S);
        std::copy(x.begin(), x.end(), res.x_filt.begin() + std::size_t(t) * d);
        std::copy(P.begin(), P.end(), res.P_filt.begin() + std::size_t(t) * d * d);
    }

    // Rauch-Tung-Striebel smoother.
    res.x_smooth = res.x_filt;
    res.P_smooth = res.P_filt;
    std::vector<double> G(d * d), PfAt(d * d), diff(d), Pd(d * d), Gt(d * d);
    for (int t = n - 2; t >= 0; --t) {
        const double* Pf = &res.P_filt[std::size_t(t) * d * d];
        const double* xf = &res.x_filt[std::size_t(t) * d];
        const double* Ppn = &P_pred_all[std::size_t(t + 1) * d * d];
        const double* xpn = &x_pred_all[std::size_t(t + 1) * d];
        double* xs = &res.x_smooth[std::size_t(t) * d];
        double* Ps = &res.P_smooth[std::size_t(t) * d * d];
        const double* xs1 = &res.x_smooth[std::size_t(t + 1) * d];
        const double* Ps1 = &res.P_smooth[std::size_t(t + 1) * d * d];

        // G = Pf A' (P_pred_{t+1})^{-1}
        detail::matmul(Pf, At.data(), PfAt.data(), d, d, d);
        std::vector<double> Ppn_j(Ppn, Ppn + d * d);
        for (int i = 0; i < d; ++i) Ppn_j[i * d + i] += 1e-12; // jitter
        detail::solve_right(Ppn_j.data(), PfAt.data(), G.data(), d, d);

        for (int i = 0; i < d; ++i) diff[i] = xs1[i] - xpn[i];
        std::vector<double> Gd(d);
        detail::matmul(G.data(), diff.data(), Gd.data(), d, d, 1);
        for (int i = 0; i < d; ++i) xs[i] = xf[i] + Gd[i];

        for (int i = 0; i < d * d; ++i) Pd[i] = Ps1[i] - Ppn[i];
        detail::matmul(G.data(), Pd.data(), tmp.data(), d, d, d);
        detail::transpose(G.data(), Gt.data(), d, d);
        detail::matmul(tmp.data(), Gt.data(), Pd.data(), d, d, d);
        for (int i = 0; i < d * d; ++i) Ps[i] = Pf[i] + Pd[i];
    }
    return res;
}

// Convenience builders -------------------------------------------------------

// Local level: x_t = x_{t-1} + w (Q=q), y_t = x_t + v (R=r).
inline KalmanSpec local_level(double y0, double q, double r) {
    KalmanSpec s;
    s.dim = 1;
    s.A = {1.0};
    s.C = {1.0};
    s.Q = {q};
    s.R = r;
    s.x0 = {y0};
    s.P0 = {r * 10.0 + 1e-8};
    return s;
}

// Local linear trend: state [level, drift]; level_t = level_{t-1} + drift_{t-1},
// drift follows a random walk. Observation sees the level only.
inline KalmanSpec local_trend(double y0, double q_level, double q_drift, double r) {
    KalmanSpec s;
    s.dim = 2;
    s.A = {1.0, 1.0, 0.0, 1.0};
    s.C = {1.0, 0.0};
    s.Q = {q_level, 0.0, 0.0, q_drift};
    s.R = r;
    s.x0 = {y0, 0.0};
    s.P0 = {r * 10.0 + 1e-8, 0.0, 0.0, r + 1e-8};
    return s;
}

} // namespace qs
