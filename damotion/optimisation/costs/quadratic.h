#ifndef COSTS_QUADRATIC_H
#define COSTS_QUADRATIC_H

#include "damotion/common/polynomial_function.h"
#include "damotion/optimisation/costs/base.h"

namespace damotion {
namespace optimisation {

/**
 * @brief A cost of the form 0.5 x^T A x + b^T x + c
 *
 */
class QuadraticCost : public Cost, public common::PolynomialFunction {
 public:
  using UniquePtr = std::unique_ptr<QuadraticCost>;
  using SharedPtr = std::shared_ptr<QuadraticCost>;

  QuadraticCost(const std::string &name,
                const common::Function::SharedPtr &fcon,
                const Bounds::Type &bounds)
      : Constraint(name, fcon, bounds), common::PolynomialFunction(2) {}

  QuadraticCost(const std::string &name, const casadi::SX &A,
                const casadi::SX &b, const casadi::SX &c, const casadi::SX &x,
                const casadi::SXVector &p)
      : Cost(mtimes(mtimes(x.T(), A), x) + mtimes(b.T(), x) + c, {x}, p, false)
      : common::PolynomialFunction(2) {
    // Compute coefficients function
    common::Function::SharedPtr fc =
        std::make_shared<damotion::casadi::FunctionWrapper>(
            ::casadi::Function(name + "quadratic_coefficients", p, {A, b, c}));
    setCoefficientsFunction(fc);
  }

  common::Function::Output &A() {
    return getCoefficientsFunction()->getOutput(0);
  }
  common::Function::Output &b() {
    return getCoefficientsFunction()->getOutput(1);
  }
  common::Function::Output &c() {
    return getCoefficientsFunction()->getOutput(2);
  }

 private:
};

}  // namespace optimisation
}  // namespace damotion

#endif /* COSTS_QUADRATIC_H */
