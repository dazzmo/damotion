#ifndef COSTS_LINEAR_H
#define COSTS_LINEAR_H

#include "damotion/optimisation/costs/base.h"

namespace damotion {
namespace optimisation {

/**
 * @brief Linear cost of the form \f$ c(x, p) = c^T(p) x + b(p) \in \mathbb{R}
 * \f$.
 *
 */
class LinearCost : public Cost {
 public:
  using UniquePtr = std::unique_ptr<LinearCost>;
  using SharedPtr = std::shared_ptr<LinearCost>;

  virtual void coeffs(OptionalVector c, double &b) const = 0;

  LinearCost(const String &name, const Index &nx, const Index &np = 0)
      : Cost(name, nx, np) {
    // Initialise coefficient matrices
    c_ = Eigen::VectorXd::Zero(this->nx());
    b_ = 0.0;
  }

  ReturnType evaluate(const InputVectorType &x,
                      OptionalJacobianType grd = nullptr) const {
    // Compute A and b
    coeffs(c_, b_);
    // Copy jacobian
    if (grd) grd = c_;
    // Compute linear constraint
    return c_.dot(x) + b_;
  }

 private:
  mutable Eigen::VectorXd c_;
  mutable double b_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* COSTS_LINEAR_H */
