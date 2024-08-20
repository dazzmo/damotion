#ifndef CONSTRAINTS_LINEAR_H
#define CONSTRAINTS_LINEAR_H

#include "damotion/optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

/**
 * @brief Linear constraint of the form \f$ c(x, p) = A(p) x + b(p) \f$.
 *
 */
class LinearConstraint : public Constraint {
 public:
  using UniquePtr = std::unique_ptr<LinearConstraint>;
  using SharedPtr = std::shared_ptr<LinearConstraint>;

  virtual void coeffs(OptionalMatrix A = nullptr,
                      OptionalVector b = nullptr) const {}

  LinearConstraint(const String &name, const Index &nc, const Index &nx)
      : Constraint(name, nc, nx) {
    // Initialise coefficient matrices
    A_ = Eigen::MatrixXd::Zero(this->nc(), this->nx());
    b_ = Eigen::VectorXd::Zero(this->nc());
  }

  ReturnType evaluate(const InputVectorType &x,
                      OptionalMatrix J = nullptr) const {
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

#endif /* CONSTRAINTS_LINEAR_H */
