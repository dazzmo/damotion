#ifndef COSTS_LINEAR_H
#define COSTS_LINEAR_H

#include "damotion/optimisation/costs/base.h"

namespace damotion {
namespace optimisation {

/**
 * @brief Cost of the form \f$ c^T x + b \f$
 *
 */
class LinearCost : public Cost {
 public:
  using UniquePtr = std::unique_ptr<LinearCost>;
  using SharedPtr = std::shared_ptr<LinearCost>;

  virtual void coeffs(OptionalJacobianType c = nullptr, const double &b = 0.0) {
  }

  LinearCost(const std::string &name) : Cost(name) {}

  ReturnType evaluate(const InputVectorType &x,
                      OptionalJacobianType g = nullptr) {
    // Compute A and b
    coeffs(c_, b_);
    // Copy jacobian
    if (g) g = c_;
    // Compute linear cost
    return c_.dot(x) + b_;
  }

 private:
  JacobianType c_;
  double b_;
};

}  // namespace optimisation
}  // namespace damotion

#endif/* COSTS_LINEAR_H */
