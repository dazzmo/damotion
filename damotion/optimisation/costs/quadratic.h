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

  virtual void coeffs(OptionalHessianType A, OptionalVectorType b,
                      double &c) const = 0;

  QuadraticCost(const String &name, const Index &nx, const Index &np = 0)
      : Cost(name, nx, np) {
    A_ = HessianType::Zero(nx, nx);
    b_ = JacobianType::Zero(1, nx);
    c_ = 0.0;
  }

  ReturnType evaluate(const InputVectorType &x,
                      OptionalJacobianType g = nullptr) const {
    // Compute A and b
    coeffs(A_, b_, c_);
    // Copy jacobian
    if (g) *g = A_ * x + b_;
    // Compute linear cost
    return 0.5 * x.transpose() * A_ * x + b_.dot(x) + c_;
  }

 private:
  mutable HessianType A_;
  mutable VectorType b_;
  mutable double c_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* COSTS_QUADRATIC_H */
