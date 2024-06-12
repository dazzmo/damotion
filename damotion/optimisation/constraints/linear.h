#ifndef CONSTRAINTS_LINEAR_H
#define CONSTRAINTS_LINEAR_H

#include "damotion/optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

class LinearConstraint : public Constraint {
 public:
  using UniquePtr = std::unique_ptr<LinearConstraint>;
  using SharedPtr = std::shared_ptr<LinearConstraint>;

  LinearConstraint(const std::string &name, common::Function::SharedPtr &f,
                   common::Function::SharedPtr &fc,
                   common::Function::SharedPtr &fb, const BoundsType &bounds)
      : Constraint(name, f, bounds), fA_(std::move(fA)), fb_(std::move(fb)) {
    // Create constraint based on the function
  }

  LinearConstraint(const std::string &name, const ::casadi::SX &A,
                   const ::casadi::SX &b, const ::casadi::SX &x,
                   const ::casadi::SXVector &p, const BoundsType &bounds)
      : Constraint(name, mtimes(A, x) + b, ::casadi::SXVector({x}), p, bounds) {
    // Create constraint based on the function

    // Create functions for A and b seperately
    ::casadi::Function f_coefs("lin_coefs", p, {A, b});
    f_coefs_ = std::make_shared<damotion::casadi::FunctionWrapper>(f_coefs);
  }

  /**
   * @brief Evaluates the coefficients comprising the linear constraint (i.e. A,
   * b)
   *
   * @param input
   */
  void EvalCoefficients(const common::InputRefVector &p) { f_coefs_->Eval(p); }

  /**
   * @brief The coefficient matrix A for the expression A x + b.
   *
   * @return const GenericEigenMatrix&
   */
  const GenericEigenMatrix &A() const {
    return f_coefs_->GetOutput(CoefficientIndices::A);
  }

  /**
   * @brief The constant vector for the linear constraint A x + b
   *
   * @return const GenericEigenMatrix&
   */
  const GenericEigenMatrix &b() const {
    return f_coefs_->GetOutput(CoefficientIndices::b);
  }

 private:
  const enum CoefficientIndices { A = 0, b };
  mutable common::Function::SharedPtr f_coefs_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_LINEAR_H */
