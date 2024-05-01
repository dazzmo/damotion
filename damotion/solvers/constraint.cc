#include "damotion/solvers/constraint.h"

namespace damotion {
namespace optimisation {

Constraint::Constraint(const std::string &name, const symbolic::Expression &c,
                       const BoundsType &bounds, bool jac, bool hes)
    : Constraint(name, "constraint") {
  // Resize the constraint
  Resize(c.size1(), c.Variables().size(), c.Parameters().size());

  // Create functions to compute the constraint and derivatives given the
  // variables and parameters
  casadi::SXVector in = c.Variables();
  for (const casadi::SX &pi : c.Parameters()) {
    in.push_back(pi);
  }

  // Constraint
  SetConstraintFunction(casadi::Function(name_, in, {c}));

  // Jacobian
  if (jac) {
    casadi::SXVector jacobians;
    for (const casadi::SX &xi : c.Variables()) {
      jacobians.push_back(jacobian(c, xi));
    }
    // Wrap the functions
    SetJacobianFunction(casadi::Function(name_ + "_jac", in, jacobians));
  }

  if (hes) {
    // TODO - Hessians
  }

  // Update bounds for the constraint
  UpdateBounds(bounds);
}

Constraint::Constraint(const std::string &name,
                       const std::string &constraint_type) {
  // Set default name for constraint
  if (name != "") {
    name_ = name;
  } else {
    name_ = constraint_type + "_" + std::to_string(CreateID());
  }
}

}  // namespace optimisation
}  // namespace damotion
