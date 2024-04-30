#include "solvers/solver.h"

namespace damotion {
namespace optimisation {
namespace solvers {

void Solver::EvaluateCost(Binding<CostType>& binding, const Eigen::VectorXd& x,
                          bool grd, bool hes, bool update_cache) {
    common::InputRefVector x_in = {}, p_in = {};
    // Extract the input for the binding
    BindingInputData& binding_input_data = GetBindingInputData(binding);
    UpdateBindingInputData(binding, x, binding_input_data);

    // Set variables
    for (int i = 0; i < binding.NumberOfVariables(); ++i) {
        x_in.push_back(binding_input_data.inputs[i]);
    }
    // Set parameters
    for (int i = 0; i < binding.NumberOfParameters(); ++i) {
        p_in.push_back(
            GetCurrentProgram().GetParameterValues(binding.GetParameter(i)));
    }

    const CostType& cost = binding.Get();
    // Evaluate the constraint
    cost.eval(x_in, p_in, grd);
    // if(hes) constraint.eval_hessian(x_in, p_in, l_in)

    if (update_cache == false) return;

    objective_cache_ += cost.ObjectiveFunction()->getOutput(0);
    if (grd) UpdateCostGradient(binding, binding_input_data);
    if (hes) UpdateLagrangianHessian(binding, binding_input_data);
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
void Solver::EvaluateConstraint(Binding<ConstraintType>& binding,
                                const int& constraint_idx,
                                const Eigen::VectorXd& x, bool jac,
                                bool update_cache) {
    common::InputRefVector x_in = {}, p_in = {};

    BindingInputData& binding_input_data = GetBindingInputData(binding);
    UpdateBindingInputData(binding, x, binding_input_data);

    // Set variables
        x_in.push_back(binding_input_data.inputs[i]);
    // Set parameters
    for (int i = 0; i < binding.NumberOfParameters(); ++i) {
        p_in.push_back(
            GetCurrentProgram().GetParameterValues(binding.GetParameter(i)));
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

    UpdateConstraintJacobian(binding, binding_input_data, constraint_idx);
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
                                      const BindingInputData& data,
                                      const int& constraint_idx) {
        Eigen::Block<Eigen::MatrixXd> J = constraint_jacobian_cache_.middleRows(
            constraint_idx, binding.Get().Dimension());

        const sym::VariableVector& x = binding.GetVariableVector();
        if (data.continuous) {
            J.middleCols(GetCurrentProgram().GetDecisionVariableIndex(vi[0]),
                         vi.size()) += J;
        } else {
            // For each variable, update the location in the Jacobian
            for (int j = 0; j < vi.size(); ++j) {
                int idx = GetCurrentProgram().GetDecisionVariableIndex(vi[j]);
                constraint_jacobian_cache_.col(idx) += J.col(j);
            }
        }
}

void Solver::UpdateLagrangianHessian(const Binding<CostType>& binding,
                                     const BindingInputData& data) {
    for (int i = 0; i < binding.NumberOfVariables(); ++i) {
        const sym::VariableVector& vi = binding.GetVariable(i);
        int i_sz = vi.size();
        int i_idx = GetCurrentProgram().GetDecisionVariableIndex(vi[0]);
        for (int j = i; j < binding.NumberOfVariables(); ++j) {
            const sym::VariableVector& vj = binding.GetVariable(j);
            // For each variable combination
            if (data.continuous[i] && data.continuous[j]) {
                int j_sz = vj.size();
                int j_idx = GetCurrentProgram().GetDecisionVariableIndex(vj[0]);
                // Create lower triangular Hessian
                if (i_idx > j_idx) {
                    lagrangian_hes_cache_.block(i_idx, j_idx, i_sz, j_sz) +=
                        binding.Get().Hessian(i, j);
                } else {
                    lagrangian_hes_cache_.block(j_idx, i_idx, j_sz, i_sz) +=
                        binding.Get().Hessian(i, j).transpose();
                }

            } else {
                // For each variable pair, populate the Hessian
                for (int ii = 0; ii < vi.size(); ++ii) {
                    for (int jj = 0; jj < vj.size(); ++jj) {
                        int i_idx =
                                GetCurrentProgram().GetDecisionVariableIndex(
                                    vi[ii]),
                            j_idx =
                                GetCurrentProgram().GetDecisionVariableIndex(
                                    vj[jj]);
                        // Create lower triangular matrix
                        if (i_idx > j_idx) {
                            lagrangian_hes_cache_(i_idx, j_idx) +=
                                binding.Get().Hessian(i, j)(ii, jj);
                        } else {
                            lagrangian_hes_cache_(j_idx, i_idx) +=
                                binding.Get().Hessian(i, j)(ii, jj);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion