#ifndef COLLOCATION_BASE_H
#define COLLOCATION_BASE_H

#include <casadi/casadi.hpp>

#include "damotion/optimisation/constraints/base.h"
#include "damotion/symbolic/expression.h"

namespace damotion {
namespace planning {
namespace optimisation {

class CollocationConstraintBase {
 public:
  CollocationConstraintBase() = default;
  ~CollocationConstraintBase() = default;

  // CollocationConstraintBase(const &model)
  template <typename MatrixType>
  std::shared_ptr<damotion::optimisation::ConstraintBase<MatrixType>>
  GetConstraint() {
    auto c = std::make_shared<ConstraintBase<MatrixType>>(
        "collocation", con_, damotion::optimisation::BoundsType::kEquality,
        true, true);
    return c;
  }

 protected:
  symbolic::Expression &GetConstraintExpression() { return con_; }

 private:
  int nx_ = 0;
  int nu_ = 0;

  symbolic::Expression con_;
};

class TrapezoidalCollocationConstraint : public CollocationConstraintBase {
 public:
  TrapezoidalCollocationConstraint(int nx, int ndx, int nu, const casadi::SX &f,
                                   const casadi::SX &x, const casadi::SX &u) {
    // Evaluate the constraint
    casadi::SX x0 = casadi::SX::sym("x0", nx), x1 = casadi::SX::sym("x1", nx);
    casadi::SX u0 = casadi::SX::sym("u0", nu), u1 = casadi::SX::sym("u1", nu);
    // Evaluate function at selected points
    casadi::SX f0 = casadi::SX::substitute(f, vertcat(x, u), vertcat(x0, u0)),
               f1 = casadi::SX::substitute(f, vertcat(x, u), vertcat(x1, u1));
    casadi::SX h = casadi::SX::sym("h", 1);

    // Integrate over interval to provide collocation constraint
    GetConstraintExpression() = x1 - x0 - 0.5 * h * (f0 + f1);
    GetConstraintExpression().SetInputs({x0, x1, u0, u1}, {});
  }

 private:
};

class HermiteSimpsonCollocationConstraint {};

}  // namespace optimisation
}  // namespace planning
}  // namespace damotion

#endif /* COLLOCATION_BASE_H */
