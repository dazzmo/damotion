#ifndef COSTS_QUADRATIC_H
#define COSTS_QUADRATIC_H

#include "damotion/optimisation/costs/base.h"

namespace damotion {
namespace optimisation {

/**
 * @brief A cost of the form 0.5 x^T A x + b^T x + c
 *
 */
class QuadraticCost : public Cost {
 public:
  using UniquePtr = std::unique_ptr<QuadraticCost>;
  using SharedPtr = std::shared_ptr<QuadraticCost>;

  virtual void coeffs(OptionalHessianType A = nullptr,
                      OptionalJacobianType b = nullptr, const double &c = 0.0) {
  }

  QuadraticCost(const std::string &name) : Cost(name) {}

  ReturnType evaluate(const InputVectorType &x,
                      OptionalJacobianType g = nullptr) {
    // Compute A and b
    coeffs(A_, b_, c_);
    // Copy jacobian
    if (g) *g = A_ * x + b_;
    // Compute linear cost
    return 0.5 * x.transpose() * A_ * x + b_.dot(x) + c_;
  }

 private:
  HessianType A_;
  JacobianType b_;
  double c_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* COSTS_QUADRATIC_H */
