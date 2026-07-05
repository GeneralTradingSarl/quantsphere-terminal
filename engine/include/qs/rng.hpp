#pragma once
// QuantSphere Engine — deterministic parallel RNG.
// Work is split into fixed WORK_UNITS chunks, each seeded independently via
// splitmix64, so results are bit-reproducible regardless of thread count.

#include <cstdint>
#include <random>

namespace qs {

inline constexpr int WORK_UNITS = 64;

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

class Rng {
public:
    Rng(std::uint64_t seed, std::uint64_t stream)
        : gen_(splitmix64(seed ^ splitmix64(stream + 1))) {}

    double normal() { return normal_(gen_); }
    double uniform() { return uniform_(gen_); }

    // Poisson with small means (lambda*dt); std::poisson_distribution is fine.
    int poisson(double mean) {
        std::poisson_distribution<int> d(mean);
        return d(gen_);
    }

    // Gamma(shape, scale). Used by the Variance-Gamma subordinator.
    double gamma(double shape, double scale) {
        std::gamma_distribution<double> d(shape, scale);
        return d(gen_);
    }

private:
    std::mt19937_64 gen_;
    std::normal_distribution<double> normal_{0.0, 1.0};
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};
};

} // namespace qs
