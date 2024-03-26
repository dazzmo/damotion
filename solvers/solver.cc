#include "solvers/solver.h"

namespace damotion {
namespace optimisation {
namespace solvers {

SolverBase::SolverBase(Program& prog) : prog_(prog) {
    // Construct caches
    // ! Program is currently dense, look at sparse alternative soon

    // Decision variables
    decision_variable_cache_ =
        Eigen::VectorXd::Zero(prog_.NumberOfDecisionVariables());
    // Output solution vector
    primal_solution_x_ =
        Eigen::VectorXd::Zero(prog_.NumberOfDecisionVariables());
    // Objective gradient
    objective_gradient_cache_ =
        Eigen::VectorXd::Zero(prog_.NumberOfDecisionVariables());
    // Constraint vector
    constraint_cache_ = Eigen::VectorXd::Zero(prog_.NumberOfConstraints());
    // Dual variables
    lambda_cache_ = Eigen::VectorXd::Zero(prog_.NumberOfConstraints());
    // Dense constraint Jacobian
    constraint_jacobian_cache_ = Eigen::MatrixXd::Zero(
        prog_.NumberOfConstraints(), prog_.NumberOfDecisionVariables());

    // Dense linearised Jacobian component
    constraint_linearised_b_cache_ =
        Eigen::VectorXd::Zero(prog_.NumberOfConstraints());

    // Dense constraint Hessian
    lagrangian_hes_cache_ = Eigen::MatrixXd::Zero(
        prog_.NumberOfDecisionVariables(), prog_.NumberOfDecisionVariables());
}

void SolverBase::EvaluateCost(Cost& cost, 
                              const Eigen::VectorXd& x, bool grad,
                              bool hes) {
    // Get cost ID
    
    // Map variables
    cost.ObjectiveFunction().call();
    objective_cache_ +=
        cost.weighting() * cost.ObjectiveFunction().getOutput(0)(0);

    if (grad) {
        // Evaluate the gradient of the objective
        cost.GradientFunction().call();
        // Set gradient blocks
        for (int i = 0; i < cost.GradientFunction().n_out(); ++i) {
            BlockIndex& idx = ;
            objective_gradient_cache_.middleRows(idx.i_start(), idx.i_sz()) +=
                cost.weighting() * cost.GradientFunction().getOutput(i);
        }
    }

    if (hes) {
        // Evaluate the hessian of the objective
        cost.HessianFunction().call();
        // Set hessian blocks
        for (int i = 0; i < cost.HessianFunction().n_out(); ++i) {
            BlockIndex& idx = cost.GetHessianBlockIndex(i);
            lagrangian_hes_cache_.block(idx.i_start(), idx.j_start(),
                                        idx.i_sz(), idx.j_sz()) +=
                cost.weighting() * cost.HessianFunction().getOutput(i);
        }
    }
}

void SolverBase::EvaluateCosts(const Eigen::VectorXd& x, bool grad, bool hes) {
    // Update program decision variables
    prog_.SetDecisionVariablesFromVector(x);

    // Reset objective
    objective_cache_ = 0.0;
    // Reset objective gradient
    if (grad) {
        objective_gradient_cache_.setZero();
    }
    // Loop through all constraints
    for (Cost& c : prog_.GetCosts()) {
        EvaluateCost(c, x, grad, hes);
    }
}

// Evaluates the constraint and updates the cache for the gradients
void SolverBase::EvaluateConstraint(Constraint& c, const Eigen::VectorXd& x,
                                    bool jac) {
    // Evaluate the constraint
    c.ConstraintFunction().call();
    // Add to constraint cache
    BlockIndex& idx = c.idx();

    constraint_cache_.middleRows(idx.i_start(), idx.i_sz()) =
        c.ConstraintFunction().getOutput(0);

    if (jac) {
        // c.LinearisedConstraintFunction().call();
        c.JacobianFunction().call();

        // ! Currently a dense jacobian
        // Update Jacobian blocks
        for (int i = 0; i < c.JacobianFunction().n_out(); ++i) {
            BlockIndex& idx = c.GetJacobianBlockIndex(i);
            constraint_jacobian_cache_.block(idx.i_start(), idx.j_start(),
                                             idx.i_sz(), idx.j_sz()) =
                c.JacobianFunction().getOutput(i);
        }

        // Add Jacobian and constant component
        // constraint_linearised_b_cache_.middleRows(c.idx(), c.dim())
        // << c.LinearisedConstraintFunction().getOutput(1);
    }
}

void SolverBase::EvaluateConstraints(const Eigen::VectorXd& x, bool jac) {
    // Update program decision variables
    prog_.SetDecisionVariablesFromVector(x);

    // Reset constraint vector
    constraint_cache_.setZero();
    // Reset objective gradient
    if (jac) {
        constraint_jacobian_cache_.setZero();
    }
    // Loop through all constraints
    for (Constraint& c : prog_.GetConstraints()) {
        EvaluateConstraint(c, x, jac);
    }
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion