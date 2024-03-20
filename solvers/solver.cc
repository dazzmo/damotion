#include "solvers/solver.h"

namespace damotion {
namespace solvers {

SolverBase::SolverBase(Program& prog) : prog_(prog) {
    // Construct caches
    // ! Program is currently dense, look at sparse alternative soon

    decision_variable_cache_ =
        Eigen::VectorXd::Zero(prog_.NumberOfDecisionVariables());
    primal_solution_x_ =
        Eigen::VectorXd::Zero(prog_.NumberOfDecisionVariables());
    objective_gradient_cache_ =
        Eigen::VectorXd::Zero(prog_.NumberOfDecisionVariables());
    constraint_cache_ = Eigen::VectorXd::Zero(prog_.NumberOfConstraints());
    lambda_cache_ = Eigen::VectorXd::Zero(prog_.NumberOfConstraints());
    constraint_jacobian_cache_ = Eigen::MatrixXd::Zero(
        prog_.NumberOfConstraints(), prog_.NumberOfDecisionVariables());

    lagrangian_hes_cache_ = Eigen::MatrixXd::Zero(
        prog_.NumberOfDecisionVariables(), prog_.NumberOfDecisionVariables());
}

void SolverBase::EvaluateCost(Program::Cost& cost, const Eigen::VectorXd& x,
                              bool grad, bool hes) {
    // Map variables
    cost.obj.call();
    objective_cache_ += cost.weighting() * cost.obj.getOutput(0)(0);

    if (grad) {
        // Evaluate the gradient of the objective
        cost.grad.call();
        objective_gradient_cache_ += cost.weighting() * cost.grad.getOutput(0);
    }

    if (hes) {
        // Evaluate the hessian of the objective
        cost.hes.call();
        lagrangian_hes_cache_ += cost.weighting() * cost.hes.getOutput(0);
    }
}

void SolverBase::EvaluateCosts(const Eigen::VectorXd& x, bool grad, bool hes) {
    // Update program decision variables
    prog_.UpdateDecisionVariables(x);

    // Reset objective
    objective_cache_ = 0.0;
    // Reset objective gradient
    if (grad) {
        objective_gradient_cache_.setZero();
    }
    // Loop through all constraints
    for (auto& c : prog_.Costs()) {
        EvaluateCost(c.second, x, grad, hes);
    }
}

// Evaluates the constraint and updates the cache for the gradients
void SolverBase::EvaluateConstraint(Program::Constraint& c,
                                    const Eigen::VectorXd& x, bool jac) {
    // Update variables
    prog_.UpdateDecisionVariables(x);

    // Evaluate the constraint
    c.con.call();
    // Add to constraint cache
    constraint_cache_.middleRows(c.idx(), c.dim()) = c.con.getOutput(0);

    if (jac) {
        c.jac.call();
        // ! Currently a dense jacobian
        constraint_jacobian_cache_.middleRows(c.idx(), c.dim())
            << c.jac.getOutput(0);
    }
}

void SolverBase::EvaluateConstraints(const Eigen::VectorXd& x, bool jac) {
    // Update program decision variables
    prog_.UpdateDecisionVariables(x);

    // Reset constraint vector
    constraint_cache_.setZero();
    // Reset objective gradient
    if (jac) {
        constraint_jacobian_cache_.setZero();
    }
    // Loop through all constraints
    for (auto& c : prog_.Constraints()) {
        EvaluateConstraint(c.second, x, jac);
    }
}

}  // namespace optimisation
}  // namespace damotion