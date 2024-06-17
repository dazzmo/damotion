#ifndef CONSTRAINTS_LINEAR_H
#define CONSTRAINTS_LINEAR_H

#include "damotion/optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

class LinearConstraint : public Constraint {
 public:
  using UniquePtr = std::unique_ptr<LinearConstraint>;
  using SharedPtr = std::shared_ptr<LinearConstraint>;

  LinearConstraint(const std::string &name,
                   const common::Function::SharedPtr &fcon,
                   const BoundsType &bounds,
                   const common::Function::SharedPtr &fcoefs = nullptr)
      : Constraint(name, fcon, bounds), fcoefs_(std::move(fcoefs)) {
    // Create constraint based on the function
  }

  LinearConstraint(const std::string &name, const ::casadi::SX &A,
                   const ::casadi::SX &b, const ::casadi::SX &x,
                   const ::casadi::SXVector &p, const BoundsType &bounds)
      : Constraint(name, mtimes(A, x) + b, ::casadi::SXVector({x}), p, bounds) {
    // Create functions for A and b seperately
    ::casadi::Function f_coefs("lin_coefs", p, {A, b});
    fcoefs_ = std::make_shared<damotion::casadi::FunctionWrapper>(f_coefs);
  }

  bool hasCoefficientFunction() { return fcoefs_ != nullptr; }

  /**
   * @brief Evaluates the coefficients comprising the linear constraint (i.e. A,
   * b)
   *
   * @param p
   */
  void evalCoefficients(const std::vector<ConstVectorRef> &p,
                        std::vector<MatrixRef> &coefs) {
    fcoefs_->eval(p, coefs);
  }

 private:
  // Optional function to compute constraints individually
  mutable common::Function::SharedPtr fcoefs_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_LINEAR_H */
