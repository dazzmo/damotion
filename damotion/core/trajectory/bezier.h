#ifndef TRAJECTORY_BEZIER_H
#define TRAJECTORY_BEZIER_H

#include <memory>

#include "damotion/core/math/binomial.h"
#include "damotion/core/math/random.h"
#include "damotion/core/trajectory/trajectory.h"

namespace damotion {
namespace trajectory {

template <typename Scalar>
class Bezier : public Trajectory<Scalar> {
 public:
  Bezier(int n, int order) : Trajectory<Scalar>(n) {
    // this->dimension of curve
    order_ = order;
    // Duration of curve
    this->UpdateBeginningTime(0.0);
    this->UpdateEndingTime(1.0);
    // Control points and derivative calculations
    P_.resize(order + 1, Eigen::VectorX<Scalar>::Zero(n));
    D_.resize(order + 1, Eigen::VectorX<Scalar>::Zero(n));
  }

  /**
   * @brief Order of the Bezier curve
   *
   * @return const int&
   */
  const int& order() const { return order_; }

  std::vector<Eigen::VectorX<Scalar>>& ControlPoints() { return P_; }

  /**
   * @brief Sets the initial gradient of the curve to a desired value
   *
   * @param gradient
   */
  void SetInitialGradient(const Eigen::VectorXd& gradient) {
    assert(order_ > 1 && "Order of the curve is not high enough");
    P_[1] = P_[0] + (1.0 / order()) * gradient;
  }

  Eigen::VectorX<Scalar> eval(const Scalar& t) {
    // Evaluate the curve at this point
    Eigen::VectorX<Scalar> res = Eigen::VectorX<Scalar>::Zero(this->dim());
    // Scale t to [start, end]
    Scalar tau = (t - this->t0()) / (this->tf() - this->t0());
    for (int i = 0; i <= order_; ++i) {
      res += BernsteinPolynomial(tau, i, order_) * P_[i];
    }
    return res;
  }

  Eigen::VectorX<Scalar> derivative(const Scalar& t, int order) {
    // Compute the finite differencing coefficients
    ComputeDerivativeCoefficients(order);
    // Evaluate the curve at this point
    Eigen::VectorX<Scalar> res = Eigen::VectorX<Scalar>::Zero(this->dim());
    Scalar tau = (t - this->t0()) / (this->tf() - this->t0());
    for (int i = 0; i <= order_ - order; ++i) {
      res += BernsteinPolynomial(tau, i, order_ - order) * D_[i];
    }
    return res;
  }

 private:
  int order_;
  // Bezier control points
  std::vector<Eigen::VectorX<Scalar>> P_;
  // Finite difference coefficients
  std::vector<Eigen::VectorX<Scalar>> D_;

  // Compute the derivative coefficients
  void ComputeDerivativeCoefficients(int order) {
    // Copy control points
    for (int i = 0; i <= order_; i++) D_[i] = P_[i];
    // Compute differences
    for (int i = 0; i < order; i++) {
      for (int j = 0; j < order_ - i; j++) {
        D_[j] = (order_ - i) * (D_[j + 1] - D_[j]) / (this->tf() - this->t0());
      }
    }
  }

  /**
   * @brief Computes a Bernstein polynomial
   *
   * @param t
   * @param i
   * @param n
   * @return double
   */
  Scalar BernsteinPolynomial(const Scalar& t, int i, int n) {
    // Normalise coordinate to [0, 1]
    return math::BinomialCoefficient(n, i) * pow(1.0 - t, n - i) * pow(t, i);
  }
};

/**
 * @brief Truncates the Bezier curve between \f$ [t_0, t_f] \f$ at a point \f$ t
 * \in [t_0, t_f] \f$ such that the curve over \f$ t \in [t_0, t] \f$ is
 * returned
 *
 * @param curve
 * @param t
 * @return std::unique_ptr<Bezier<double>>
 */
std::unique_ptr<Bezier<double>> TruncateBezier(Bezier<double>& curve, double t);

void RandomiseBezier(Bezier<double>& curve, math::RandomNumberGenerator& rng,
                     double gamma = 0.1);

std::ostream& operator<<(std::ostream& os, Bezier<double>& bezier);

}  // namespace trajectory
}  // namespace damotion

#endif /* TRAJECTORY_BEZIER_H */
