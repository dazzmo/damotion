#include "solvers/constraint.h"

namespace damotion {
namespace optimisation {

Constraint::Constraint(const symbolic::Expression &c,
               const BoundsType &bounds, bool jac, bool hes) {
    // Create functions of the form f(x, p)

    // Get input sizes
    nx_ = c.Variables().size();
    np_ = c.Parameters().size();
    // Create functions to compute the constraint and derivatives given the
    // variables and parameters

    // Input vectors {x, p}
    casadi::SXVector in = c.Variables();
    for (const casadi::SX &pi : c.Parameters()) {
        in.push_back(pi);
    }

    // Create functions for each and wrap them
    // Constraint
    SetConstraintFunction(casadi::Function("con", in, {c}));

    // Jacobian
    if (jac) {
        casadi::SXVector jacobians;
        for (const casadi::SX &xi : c.Variables()) {
            jacobians.push_back(jacobian(c, xi));
        }
        // Wrap the functions
        SetJacobianFunction(casadi::Function("con_jac", in, jacobians));
    }

    if (hes) {
        // TODO - Hessians
    }

    // Set bounds for the constraint
    SetBounds(ub(), lb(), bounds);
}

}  // namespace optimisation
}  // namespace damotion