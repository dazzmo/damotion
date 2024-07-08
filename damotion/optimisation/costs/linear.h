#ifndef COSTS_LINEAR_H
#define COSTS_LINEAR_H

#include "damotion/common/polynomial_function.h"
#include "damotion/optimisation/costs/base.h"

namespace damotion {
namespace optimisation {

/**
 * @brief Cost of the form \f$ c^T x + b \f$
 *
 */
class LinearCost : public Cost, public common::PolynomialFunction {
 public:
  LinearCost(const std::string &name, const casadi::SX &c, const casadi::SX &b,
             const casadi::SX &x, const casadi::SXVector &p)
      : Cost(mtimes(c.T(), x) + b, {x}, p, false),
        common::PolynomialFunction(1) {
    // Compute coefficients function
    common::Function::SharedPtr fc =
        std::make_shared<damotion::casadi::FunctionWrapper>(
            ::casadi::Function(name + "linear_coefficients", p, {A, b}));
    setCoefficientsFunction(fc);
  }

  LinearCost(const std::string &name, const common::Function::SharedPtr &fcon,
             const Bounds::Type &bounds)
      : Constraint(name, fcon, bounds), common::PolynomialFunction(2) {}

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

#endif /* COSTS_LINEAR_H */
