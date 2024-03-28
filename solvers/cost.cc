#include "solvers/cost.h"

namespace damotion {
namespace optimisation {

Cost::Cost(const symbolic::Expression &ex, bool grd, bool hes) {
    // Get input sizes
    // Get input sizes
    nx_ = ex.Variables().size();
    np_ = ex.Parameters().size();
    // Create functions to compute the constraint and derivatives given the
    // variables and parameters

    // Input vectors {x, p}
    casadi::SXVector in = ex.Variables();
    for (const casadi::SX &pi : ex.Parameters()) {
        in.push_back(pi);
    }

    // Create functions for each and wrap them
    // Constraint
    SetObjectiveFunction(casadi::Function("cost", in, {ex}));

    // Jacobian
    if (grd) {
        casadi::SXVector gradients;
        for (const casadi::SX &xi : ex.Variables()) {
            gradients.push_back(gradient(ex, xi));
        }
        // Wrap the functions
        SetGradientFunction(casadi::Function("con_grd", in, gradients));
    }
}

}  // namespace optimisation
}  // namespace damotion