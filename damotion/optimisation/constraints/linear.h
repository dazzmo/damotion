#ifndef CONSTRAINTS_LINEAR_H
#define CONSTRAINTS_LINEAR_H

#include "damotion/optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

class LinearConstraint : public Constraint {
 public:
  using UniquePtr = std::unique_ptr<LinearConstraint>;
  using SharedPtr = std::shared_ptr<LinearConstraint>;

  virtual void coeffs(OptionalMatrix A = nullptr, OptionalVector b = nullptr) const {}

  LinearConstraint(const std::string &name) : Constraint() {}

  ReturnType evaluate(const InputVectorType &x, OptionalMatrix J = nullptr) const {
    // Compute A and b
    coeffs(A_, b_);
    // Copy jacobian
    if (J) J = A_;
    // Compute linear constraint
    return A_ * x + b_;
  }

 private:
  mutable Eigen::MatrixXd A_;
  mutable Eigen::VectorXd b_;
};

}  // namespace optimisation
}  // namespace damotion

#endif/* CONSTRAINTS_LINEAR_H */
