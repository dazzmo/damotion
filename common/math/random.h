#ifndef MATH_RANDOM_H
#define MATH_RANDOM_H

#include <random>

#include <Eigen/Core>

namespace damotion {
namespace math {

/**
 * @brief Random number generator class to provide random numbers between
 * user-defined ranges and random vectors with elements random within a
 * user-defined range
 *
 */
class RandomNumberGenerator {
   public:
    RandomNumberGenerator() {
        std::random_device dev;
        this->rng_ = std::mt19937(dev());
        this->dist_ = std::uniform_real_distribution<double>(0.0, 1.0);
    }

    void SetSeed(const int seed) { this->rng_ = std::mt19937(seed); }

    double operator()(void) { return dist_(rng_); }

    /**
     * @brief Generates a random number within the bounds of [0, 1]
     *
     * @param lower
     * @param upper
     * @return double
     */
    double RandomNumber(double lower = 0.0, double upper = 1.0) {
        double sigma = dist_(rng_);
        return lower * sigma + (1.0 - sigma) * upper;
    }

    /**
     * @brief  Generates a random vector with each entry being within the bounds
     * of [0, 1]
     *
     * @param n
     * @param lower
     * @param upper
     * @return Eigen::VectorXd
     */
    Eigen::VectorXd RandomVector(const int n, double lower = 0, double upper = 1.0) {
        Eigen::VectorXd r(n);
        for (int i = 0; i < n; ++i) {
            double sigma = dist_(rng_);
            r[i] = lower * sigma + (1.0 - sigma) * upper;
        }
        return r;
    }

   private:
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;
};

}  // namespace math
}  // namespace damotion

#endif /* MATH_RANDOM_H */
