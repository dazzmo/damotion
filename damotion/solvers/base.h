#ifndef SOLVERS_BASE_H
#define SOLVERS_BASE_H

#include <unordered_map>

#include "damotion/optimisation/block.h"
#include "damotion/optimisation/program.h"

namespace damotion {
namespace optimisation {
namespace solvers {

class SolverBase {
 public:
  SolverBase(Program& program) : program_(program) {
    int nx = program_.numberOfDecisionVariables();
    int nc = program_.NumberOfConstraints();
    // Initialise vectors
    decision_variable_cache_ = Eigen::VectorXd::Zero(nx);
    primal_solution_x_ = Eigen::VectorXd::Zero(nx);
    objective_gradient_cache_ = Eigen::VectorXd::Zero(nx);
    constraint_vector_cache_ = Eigen::VectorXd::Zero(nc);
    dual_variable_cache_ = Eigen::VectorXd::Zero(nc);

    // Create block matrix functions with the provided bindings
    objective_gradient_ = std::make_unique<BlockMatrixFunction>(
        nx, 1, BlockMatrixFunction::Type::kGradient);
    constraint_jacobian_ = std::make_unique<BlockMatrixFunction>(
        nc, nx, BlockMatrixFunction::Type::kJacobian);
    lagrangian_hessian_ = std::make_unique<BlockMatrixFunction>(
        nx, nx, BlockMatrixFunction::Type::kHessian);

    // Register all bindings
    for (auto& b : program.GetAllConstraintBindings()) {
      constraint_base_bindings_.push_back(b);
      // Add bindings to the block functions
      constraint_jacobian_->AddBinding(b, b.Get().GetDerivative(0),
                                       GetCurrentProgram());
      lagrangian_hessian_->AddBinding(b, b.Get().GetHessian(0),
                                      GetCurrentProgram());
    }
    for (auto& b : program.GetAllCostBindings()) {
      cost_base_bindings_.push_back(b);
      // Add bindings to the block functions
      objective_gradient_->AddBinding(b, b.Get().GetDerivative(0),
                                      GetCurrentProgram());
      lagrangian_hessian_->AddBinding(b, b.Get().GetHessian(0),
                                      GetCurrentProgram());
    }
  }

  ~SolverBase() {}

  /**
   * @brief Returns the current program that is being solved. Allows for
   * constraints and bounds to be modified explicitly. If the internal
   * structure of the program is to be modified, please create a new instance
   * of a solver for it
   *
   * @return Program&
   */
  Program& GetCurrentProgram() { return program_; }

  /**
   * @brief Updates the existing solver with an update program (e.g.
   * parameters or bounds have changed)
   *
   * @param program
   */
  void UpdateProgram(Program& program) { program_ = program; }

  /**
   * @brief Return the primal solution for the associated program, if solved
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd& GetPrimalSolution() const {
    return primal_solution_x_;
  }

  /**
   * @brief Status of the solver, if a solution is achieved, returns true.
   *
   * @return true
   * @return false
   */
  bool IsSolved() const { return is_solved_; }

  /**
   * @brief Returns a vector of bindings to Cost<MatrixType> for each
   * binding within the program.
   *
   * @return const std::vector<Binding<Cost>>&
   */
  const std::vector<Binding<Cost>>& GetCostBindings() const {
    return cost_base_bindings_;
  }

  /**
   * @brief Returns a vector of bindings to Constraint<MatrixType> for each
   * binding within the program. Useful for iterating through all constraint's
   * base classes.
   *
   * @return const std::vector<Binding<Constraint>>&
   */
  const std::vector<Binding<Constraint>>& GetConstraintBindings() const {
    return constraint_base_bindings_;
  }

  /**
   * @brief Returns a vector of the current values of the variables provided
   * in var. If variables are all together in the vector, we can speed this
   * process up by setting is_continuous to true.
   *
   * @param var
   * @param is_continuous
   * @return Eigen::VectorXd
   */
  Eigen::VectorXd GetDecisionVariableValues(const sym::VariableVector& var,
                                            bool is_continuous = false) {
    std::vector<int> indices =
        GetCurrentProgram().getDecisionVariableIndices(var);
    return decision_variable_cache_(indices);
  }

  /**
   * @brief All constraints within the program
   *
   * @return std::vector<Binding<Constraint>>&
   */
  std::vector<Binding<Constraint>>& GetConstraints() { return constraints_; }

  /**
   * @brief All costs within the program
   *
   * @return std::vector<Binding<Cost>>&
   */
  std::vector<Binding<Cost>>& GetCosts() { return costs_; }

  void EvaluateCost(const Binding<Cost>& binding, const Eigen::VectorXd& x,
                    bool grd, bool hes, bool update_cache = true);

  void EvaluateCosts(const Eigen::VectorXd& x, bool grd, bool hes);

  // Evaluates the constraint and updates the cache for the gradients
  void EvaluateConstraint(const Binding<Constraint>& binding,
                          const int& constraint_idx, const Eigen::VectorXd& x,
                          bool jac, bool update_cache = true);

  void EvaluateConstraints(const Eigen::VectorXd& x, bool jac);

  void UpdateConstraintJacobian(const Binding<Constraint>& binding,
                                const int& constraint_idx);

  void UpdateLagrangianHessian(const Binding<Cost>& binding);

  template <typename T>
  void EvaluateBinding(const Binding<T>& binding, const Eigen::VectorXd& x,
                       const Eigen::VectorXd& p, bool derivative,
                       bool hessian) {
    // Create vectors for each input
    std::vector<Eigen::Ref<const Eigen::VectorXd>> in;
    for (int i = 0; i < binding.GetInputSize(); ++i) {
      sym::VariableVector& xi = binding.x()[i];
      std::vector<int> indices =
          GetCurrentProgram().getDecisionVariableIndices(xi);
      // Create indexed vector view for the system
      in.push_back(x(indices));
    }

    for (int i = 0; i < binding.GetInputSize(); ++i) {
      sym::VariableVector& pi = binding.p()[i];
      std::vector<int> indices = GetCurrentProgram().getParameterIndices(pi);
      // Create indexed vector view for the system
      in.push_back(p(indices));
    }

    // Set the values for the expression
    for (int i = 0; i < in.size(); ++i) binding.Get().SetInput(i, in[i]);

    // Evaluate the binding
    binding.Get().call(derivative, hessian);
  }

 protected:
  // Cache for current values of the decision variables
  Eigen::VectorXd decision_variable_cache_;
  // Cache for current values of the dual variables
  Eigen::VectorXd dual_variable_cache_;
  // Cache for current value of the objective
  double objective_cache_;
  // Cache for current value of the objective gradient
  Eigen::VectorXd objective_gradient_cache_;
  // Cache for current value of the constraint vector
  Eigen::VectorXd constraint_vector_cache_;

  // Constraint Jacobian
  BlockMatrixFunction::UniquePtr objective_gradient_;
  BlockMatrixFunction::UniquePtr constraint_jacobian_;
  BlockMatrixFunction::UniquePtr lagrangian_hessian_;

  // Primal solution of the program
  Eigen::VectorXd primal_solution_x_;

  // Vector of the constraint bindings for the program
  std::vector<Binding<Constraint>> constraints_;
  // Vector of the cost bindings for the program
  std::vector<Binding<Cost>> costs_;

 private:
  bool is_solved_ = false;
  // Reference to current program in solver
  Program& program_;
  // Index of the constraints within the constraint vector
  std::vector<int> constraint_idx_;

  std::vector<Binding<Cost>> cost_base_bindings_ = {};
  std::vector<Binding<Constraint>> constraint_base_bindings_ = {};
};

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_BASE_H */
