#ifndef CONSTRAINTS_LINEAR_H
#define CONSTRAINTS_LINEAR_H

#include "damotion/common/polynomial_function.h"
#include "damotion/optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

class LinearConstraint : public Constraint, public common::PolynomialFunction {
 public:
  using UniquePtr = std::unique_ptr<LinearConstraint>;
  using SharedPtr = std::shared_ptr<LinearConstraint>;

  LinearConstraint(const std::string &name,
                   const common::Function::SharedPtr &fcon,
                   const Bounds::Type &bounds)
      : Constraint(name, fcon, bounds), common::PolynomialFunction(1) {}

  LinearConstraint(const std::string &name, const ::casadi::SX &A,
                   const ::casadi::SX &b, const ::casadi::SX &x,
                   const ::casadi::SXVector &p, const Bounds::Type &bounds)
      : Constraint(name, mtimes(A, x) + b, ::casadi::SXVector({x}), p, bounds),
        common::PolynomialFunction(1) {
    // Compute coefficients function
    common::Function::SharedPtr fc =
        std::make_shared<damotion::casadi::FunctionWrapper>(
            ::casadi::Function(name + "linear_coefficients", p, {A, b}));
    setCoefficientsFunction(fc);
  }

  common::Function::Output &A() {
    return getCoefficientsFunction()->getOutput(0);
  }
  common::Function::Output &b() {
    return getCoefficientsFunction()->getOutput(1);
  }

 private:
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_LINEAR_H */
