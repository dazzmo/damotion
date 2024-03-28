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
    dual_variable_cache_ = Eigen::VectorXd::Zero(prog_.NumberOfConstraints());
    // Dense constraint Jacobian
    constraint_jacobian_cache_ = Eigen::MatrixXd::Zero(
        prog_.NumberOfConstraints(), prog_.NumberOfDecisionVariables());

    // Dense constraint Hessian
    lagrangian_hes_cache_ = Eigen::MatrixXd::Zero(
        prog_.NumberOfDecisionVariables(), prog_.NumberOfDecisionVariables());
}

void SolverBase::EvaluateCost(Binding<Cost>& b, const Eigen::VectorXd& x,
                              bool grd, bool hes) {
    Cost& cost = b.Get();

    // Evaluate the objective
    for (int i = 0; i < b.NumberOfVariables(); ++i) {
        cost.ObjectiveFunction().setInput(
            i, x.data() + b.VariableStartIndices()[i]);
    }
    cost.ObjectiveFunction().call();

    // Map variables
    objective_cache_ += cost.ObjectiveFunction().getOutput(0)(0);

    if (grd) {
        // Evaluate the gradient of the objective
        for (int i = 0; i < b.NumberOfVariables(); ++i) {
            cost.GradientFunction().setInput(
                i, x.data() + b.VariableStartIndices()[i]);
        }
        cost.GradientFunction().call();

        // Add gradients of each block
        for (int i = 0; i < b.NumberOfVariables(); ++i) {
            objective_gradient_cache_.middleRows(b.VariableStartIndices()[i],
                                                 10) +=
                cost.GradientFunction().getOutput(i);
        }
    }

    if (hes) {
        // Evaluate the hessian of the objective
        // cost.HessianFunction().call();
        // // Set hessian blocks
        // for (int i = 0; i < cost.HessianFunction().n_out(); ++i) {
        //     BlockIndex& idx = cost.GetHessianBlockIndex(i);
        //     lagrangian_hes_cache_.block(idx.i_start(), idx.j_start(),
        //                                 idx.i_sz(), idx.j_sz()) +=
        //         cost.weighting() * cost.HessianFunction().getOutput(i);
        // }
    }
}

void SolverBase::EvaluateCosts(const Eigen::VectorXd& x, bool grad, bool hes) {
    // Reset objective
    objective_cache_ = 0.0;
    // Reset objective gradient
    if (grad) {
        objective_gradient_cache_.setZero();
    }
    // Evaluate all costs in the program
    for (Binding<Cost>& b : prog_.GetCostBindings()) {
        EvaluateCost(b, x, grad, hes);
    }
}

// Evaluates the constraint and updates the cache for the gradients
void SolverBase::EvaluateConstraint(Binding<Constraint>& b,
                                    const int& constraint_idx,
                                    const Eigen::VectorXd& x, bool jac) {
    Constraint& constraint = b.Get();

    // Get size of constraint
    int m = constraint.Dimension();

    // Evaluate the objective
    for (int i = 0; i < b.NumberOfVariables(); ++i) {
        constraint.ConstraintFunction().setInput(
            i, x.data() + b.VariableStartIndices()[i]);
    }
    constraint.ConstraintFunction().call();

    constraint_cache_.middleRows(constraint_idx, m) =
        constraint.ConstraintFunction().getOutput(0);

    if (jac) {
        // Evaluate the jacobians
        for (int i = 0; i < b.NumberOfVariables(); ++i) {
            constraint.JacobianFunction().setInput(
                i, x.data() + b.VariableStartIndices()[i]);
        }
        constraint.JacobianFunction().call();

        // ! Currently a dense jacobian - Look into sparse work
        // Update Jacobian blocks
        Eigen::Block<Eigen::MatrixXd> jac =
            constraint_jacobian_cache_.middleRows(constraint_idx, m);

        for (int i = 0; i < b.NumberOfVariables(); ++i) {
            jac.middleCols(b.VariableStartIndices()[i],
                           b.GetVariable(i).size()) =
                constraint.JacobianFunction().getOutput(i);
        }
    }
}

void SolverBase::EvaluateConstraints(const Eigen::VectorXd& x, bool jac) {
    // Reset constraint vector
    constraint_cache_.setZero();
    // Reset objective gradient
    if (jac) {
        constraint_jacobian_cache_.setZero();
    }
    // Loop through all constraints
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion