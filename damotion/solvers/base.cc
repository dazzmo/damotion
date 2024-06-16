#include "damotion/solvers/base.h"

namespace damotion {
namespace optimisation {
namespace solvers {

void SolverBase::EvaluateCost(const Binding<Cost>& binding,
                              const Eigen::VectorXd& x, bool grd, bool hes,
                              bool update_cache) {
  // Evaluate the binding
  EvaluateBinding(binding, x, GetCurrentProgram().ParameterVector(), grd, hes);
  if (update_cache == false) return;

  objective_cache_ += cost.Objective();
  if (grd) UpdateCostGradient(binding);
  if (hes) UpdateLagrangianHessian(binding);
}

void SolverBase::EvaluateCosts(const Eigen::VectorXd& x, bool grad, bool hes) {
  // Reset terms
  objective_cache_ = 0.0;
  if (grad) objective_gradient_cache_.setZero();
  if (hes) lagrangian_hes_cache_.setZero();

  // Evaluate all costs in the program
  for (Binding<Cost>& b : GetCosts()) {
    EvaluateCost(b, x, grad, hes);
  }
}

// Evaluates the constraint and updates the cache for the gradients
void SolverBase::EvaluateConstraint(const Binding<Constraint>& binding,
                                    const int& constraint_idx,
                                    const Eigen::VectorXd& x, bool jac,
                                    bool update_cache) {
  std::vector<ConstVectorRef> x_in = {}, p_in = {};
  GetBindingInputs(binding, x_in, p_in);

  const Constraint& constraint = binding.Get();
  // Evaluate the constraint
  constraint.eval(x_in, p_in, jac);
  // if(hes) constraint.eval_hessian(x_in, p_in, l_in)

  // Update the caches if required, otherwise break early
  if (update_cache == false) return;

  constraint_vector_cache_.middleRows(constraint_idx, constraint.dim()) =
      constraint.Vector();
  VLOG(10) << "constraint_cache = " << constraint_vector_cache_;

  UpdateConstraintJacobian(binding, constraint_idx);
  VLOG(10) << "jacobian_cache = " << constraint_jacobian_cache_;
}

void SolverBase::EvaluateConstraints(const Eigen::VectorXd& x, bool jac) {
  // Reset constraint vector
  constraint_vector_cache_.setZero();
  // Reset objective gradient
  if (jac) {
    constraint_jacobian_cache_.setZero();
  }
  // Loop through all constraints
  for (Binding<Constraint>& b : GetConstraints()) {
    EvaluateConstraint(b, 0, x, jac);
  }
}

void SolverBase::UpdateConstraintJacobian(const Binding<Constraint>& binding,
                                          const int& constraint_idx) {
  // Get data related to the binding
  BindingInputData& data = GetBindingInputData(binding);
  // Get block rows related to the binding constraint
  int idx = 0;
  for (int i = 0; i < binding.nx(); ++i) {
    const sym::VariableVector& xi = binding.x(i);
    Eigen::Ref<const Eigen::MatrixXd> Ji =
        binding.Get().Jacobian().middleCols(idx, xi.size());
    InsertJacobianAtVariableLocations(constraint_jacobian_cache_, Ji, xi,
                                      constraint_idx, data.x_continuous[i]);
    idx += xi.size();
  }
}

void SolverBase::UpdateLagrangianHessian(const Binding<Cost>& binding) {
  // Get data related to the binding
  BindingInputData& data = GetBindingInputData(binding);

  int idx_x = 0, idx_y = 0;

  for (int i = 0; i < binding.nx(); ++i) {
    const sym::VariableVector& xi = binding.x(i);
    for (int j = i; j < binding.nx(); ++j) {
      const sym::VariableVector& xj = binding.x(j);
      // Get Hessian block
      Eigen::Ref<const Eigen::MatrixXd> Hij =
          binding.Get().Hessian().block(idx_x, idx_y, xi.size(), xj.size());
      InsertHessianAtVariableLocations(lagrangian_hes_cache_, Hij, xi, xj,
                                       data.x_continuous[i],
                                       data.x_continuous[j]);

      idx_y += xj.size();
    }
    idx_x += xi.size();
  }
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion