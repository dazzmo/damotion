#include "solvers/cost.h"

namespace damotion {
namespace optimisation {

Cost::Cost(const symbolic::Expression &ex, const std::string &name, bool grd,
           bool hes) {
    // Get input sizes
    // Get input sizes
    nx_ = ex.Variables().size();
    np_ = ex.Parameters().size();

    if (name == "") {
        name_ = "obj_" + std::to_string(CreateID());
    } else {
        name_ = name;
    }

    // Create functions to compute the constraint and derivatives given the
    // variables and parameters

    // Input vectors {x, p}
    casadi::SXVector in = ex.Variables();
    for (const casadi::SX &pi : ex.Parameters()) {
        in.push_back(pi);
    }

    // Create functions for each and wrap them
    // Constraint
    SetObjectiveFunction(casadi::Function(name, in, {ex}));
    // Jacobian
    if (grd) {
        casadi::SXVector gradients;
        for (const casadi::SX &xi : ex.Variables()) {
            gradients.push_back(gradient(ex, xi));
        }
        // Wrap the functions
        SetGradientFunction(casadi::Function(name + "_grd", in, gradients));
    }

    // Hessians
    if (hes) {
        casadi::SXVector hessians;
        // For each combination of input variables, compute the hessians
        for (int i = 0; i < ex.Variables().size(); ++i) {
            casadi::SX xi = ex.Variables()[i];
            for (int j = i; j < ex.Variables().size(); ++j) {
                casadi::SX xj = ex.Variables()[j];
                if (j == i) {
                    // Diagonal term, only include lower-triangular component
                    hessians.push_back(
                        casadi::SX::tril(jacobian(gradient(ex, xi), xj), true));
                } else {
                    hessians.push_back(jacobian(gradient(ex, xi), xj));
                }
            }
        }

        // Wrap the functions
        SetHessianFunction(casadi::Function(name + "_hes", in, hessians));
    }
}

}  // namespace optimisation
}  // namespace damotion