#include "damotion/solvers/solver.h"

namespace damotion {
namespace optimisation {
namespace solvers {

void Solver::EvaluateCost(const Binding<CostType>& binding,
                          const Eigen::VectorXd& x, bool grd, bool hes,
                          bool update_cache) {
    common::InputRefVector x_in = {}, p_in = {};
    GetBindingInputs(binding, x_in);

    // Set parameters
    for (int i = 0; i < binding.np(); ++i) {
        p_in.push_back(GetCurrentProgram().GetParameterValues(binding.p(i)));
    }

    const CostType& cost = binding.Get();
    // Evaluate the constraint
    cost.eval(x_in, p_in, grd);
    // if(hes) constraint.eval_hessian(x_in, p_in, l_in)

    if (update_cache == false) return;

    objective_cache_ += cost.ObjectiveFunction()->getOutput(0);
    if (grd) UpdateCostGradient(binding);
    if (hes) UpdateLagrangianHessian(binding);
}

void Solver::EvaluateCosts(const Eigen::VectorXd& x, bool grad, bool hes) {
    // Reset terms
    objective_cache_ = 0.0;
    if (grad) objective_gradient_cache_.setZero();
    if (hes) lagrangian_hes_cache_.setZero();

    // Evaluate all costs in the program
    for (Binding<CostType>& b : GetCosts()) {
        EvaluateCost(b, x, grad, hes);
    }
}

// Evaluates the constraint and updates the cache for the gradients
void Solver::EvaluateConstraint(const Binding<ConstraintType>& binding,
                                const int& constraint_idx,
                                const Eigen::VectorXd& x, bool jac,
                                bool update_cache) {
    common::InputRefVector x_in = {}, p_in = {};
    GetBindingInputs(binding, x_in);

    // Set parameters
    for (int i = 0; i < binding.np(); ++i) {
        p_in.push_back(GetCurrentProgram().GetParameterValues(binding.p(i)));
    }

    const ConstraintType& constraint = binding.Get();
    // Evaluate the constraint
    constraint.eval(x_in, p_in, jac);
    // if(hes) constraint.eval_hessian(x_in, p_in, l_in)

    // Update the caches if required, otherwise break early
    if (update_cache == false) return;

    constraint_cache_.middleRows(constraint_idx, constraint.Dimension()) =
        constraint.Vector();
    VLOG(10) << "constraint_cache = " << constraint_cache_;

    UpdateConstraintJacobian(binding, constraint_idx);
}

void Solver::EvaluateConstraints(const Eigen::VectorXd& x, bool jac) {
    // Reset constraint vector
    constraint_cache_.setZero();
    // Reset objective gradient
    if (jac) {
        constraint_jacobian_cache_.setZero();
    }
    // Loop through all constraints
    for (Binding<ConstraintType>& b : GetConstraints()) {
        EvaluateConstraint(b, 0, x, jac);
    }
}

void Solver::UpdateConstraintJacobian(const Binding<ConstraintType>& binding,
                                      const int& constraint_idx) {
    // Get data related to the binding
    ProgramType& program = GetCurrentProgram();
    BindingInputData& data = GetBindingInputData(binding);
    Eigen::Block<Eigen::MatrixXd> J = constraint_jacobian_cache_.middleRows(
        constraint_idx, binding.Get().Dimension());

    int idx = 0;
    for (int i = 0; i < binding.nx(); ++i) {
        const sym::VariableVector& xi = binding.x(i);
        Eigen::Ref<const Eigen::MatrixXd> Ji =
            binding.Get().Jacobian().middleCols(idx, xi.size());
        InsertJacobianAtVariableLocations(constraint_jacobian_cache_, Ji, xi,
                                          data.continuous[i]);
        idx += xi.size();
    }
}

void Solver::UpdateLagrangianHessian(const Binding<CostType>& binding) {
    // Get data related to the binding
    ProgramType& program = GetCurrentProgram();
    BindingInputData& data = GetBindingInputData(binding);

    int idx_x = 0, idx_y = 0;

    for (int i = 0; i < binding.nx(); ++i) {
        const sym::VariableVector& xi = binding.x(i);
        int i_sz = xi.size();
        int i_idx = GetCurrentProgram().GetDecisionVariableIndex(xi[0]);
        for (int j = i; j < binding.nx(); ++j) {
            const sym::VariableVector& xj = binding.x(j);
            int j_sz = xj.size();
            int j_idx = GetCurrentProgram().GetDecisionVariableIndex(xj[0]);

            // Get Hessian block
            Eigen::Ref<const Eigen::MatrixXd> Hij =
                binding.Get().Hessian().block(idx_x, idx_y, i_sz, j_sz);

            InsertHessianAtVariableLocations(lagrangian_hes_cache_, Hij, xi, xj,
                                             data.continuous[j],
                                             data.continuous[j]);

            idx_y += xj.size();
        }
        idx_x += xi.size();
    }
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion