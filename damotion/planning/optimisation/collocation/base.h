#ifndef COLLOCATION_BASE_H
#define COLLOCATION_BASE_H

#include <casadi/casadi.hpp>

#include "damotion/core/logging.h"
#include "damotion/optimisation/constraints/base.h"
#include "damotion/symbolic/expression.h"

namespace damotion {
namespace planning {
namespace optimisation {

namespace casadi = ::casadi;

class CollocationConstraintBase {
 public:
  CollocationConstraintBase() = default;
  ~CollocationConstraintBase() = default;

  // CollocationConstraintBase(const &model)
  std::shared_ptr<damotion::optimisation::ConstraintBase> GetConstraint() {
    auto c = std::make_shared<damotion::optimisation::ConstraintBase>(
        "collocation", con_, damotion::optimisation::Bounds::Type::kEquality,
        true, true);
    return c;
  }

 protected:
  casadi::SX &GetConstraintExpression() { return con_; }

 private:
  int nx_ = 0;
  int nu_ = 0;

  casadi::SX con_;
  casadi::SXVector x_;
  casadi::SXVector p_;
};

class TrapezoidalCollocationConstraint : public CollocationConstraintBase {
 public:
  TrapezoidalCollocationConstraint(const int &nx, const int &ndx, const int &nu,
                                   const casadi::SX &f, const casadi::SX &x,
                                   const casadi::SX &u) {
    // Evaluate the constraint
    casadi::SX x0 = casadi::SX::sym("x0", nx), x1 = casadi::SX::sym("x1", nx);
    casadi::SX u0 = casadi::SX::sym("u0", nu), u1 = casadi::SX::sym("u1", nu);

    // Evaluate function at selected points
    casadi::SX f0 = casadi::SX::substitute(f, vertcat(x, u), vertcat(x0, u0)),
               f1 = casadi::SX::substitute(f, vertcat(x, u), vertcat(x1, u1));
    casadi::SX h = casadi::SX::sym("h", 1);

    // Integrate over interval to provide collocation constraint
    GetConstraintExpression() = x1 - x0 - 0.5 * h * (f0 + f1);
    this->x_ = {x0, x1, u0, u1, h};
    this->p_ = {};

    VLOG(10) << "Trapezoidal Collocation Constraint:";
    VLOG(10) << GetConstraintExpression();
  }

 private:
};

class HermiteSimpsonCollocationConstraint : public CollocationConstraintBase {
 public:
  HermiteSimpsonCollocationConstraint(const int &nx, const int &ndx,
                                      const int &nu, const casadi::SX &f,
                                      const casadi::SX &x,
                                      const casadi::SX &u) {
    // Evaluate the constraint
    casadi::SX x0 = casadi::SX::sym("x0", nx), xm = casadi::SX::sym("xm", nx),
               x1 = casadi::SX::sym("x1", nx);
    casadi::SX u0 = casadi::SX::sym("u0", nu), um = casadi::SX::sym("um", nu),
               u1 = casadi::SX::sym("u1", nu);

    // Evaluate function at selected points
    casadi::SX f0 = casadi::SX::substitute(f, vertcat(x, u), vertcat(x0, u0)),
               fm = casadi::SX::substitute(f, vertcat(x, u), vertcat(xm, um)),
               f1 = casadi::SX::substitute(f, vertcat(x, u), vertcat(x1, u1));
    casadi::SX h = casadi::SX::sym("h", 1);

    // Integrate over interval to provide collocation constraint
    GetConstraintExpression() =
        casadi::SX::vertcat({x1 - x0 - (h / 6.0) * (f0 + 4.0 * fm + f1),
                             xm - 0.5 * (x0 + x1) + (h / 8.0) * (f0 - f1)});
    GetConstraintExpression().SetInputs({x0, xm, x1, u0, um, u1, h}, {});

    VLOG(10) << "Hermite-Simpson Collocation Constraint:";
    VLOG(10) << GetConstraintExpression();
  }

 private:
};

}  // namespace optimisation
}  // namespace planning
}  // namespace damotion

#endif /* COLLOCATION_BASE_H */
