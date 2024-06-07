#ifndef SOLVERS_BASE_H
#define SOLVERS_BASE_H

#include <unordered_map>

#include "damotion/optimisation/program.h"

namespace damotion {
namespace optimisation {
namespace solvers {

class SolverBase {
 public:
  SolverBase(Program& program) : program_(program) {
    int nx = program_.NumberOfDecisionVariables();
    int nc = program_.NumberOfConstraints();
    // Initialise vectors
    decision_variable_cache_ = Eigen::VectorXd::Zero(nx);
    primal_solution_x_ = Eigen::VectorXd::Zero(nx);
    objective_gradient_cache_ = Eigen::VectorXd::Zero(nx);
    constraint_cache_ = Eigen::VectorXd::Zero(nc);
    dual_variable_cache_ = Eigen::VectorXd::Zero(nc);

    // Register all bindings
    for (auto& b : program.GetAllConstraintBindings()) {
      RegisterBinding(b);
      constraint_base_bindings_.push_back(b);
    }
    for (auto& b : program.GetAllCostBindings()) {
      RegisterBinding(b);
      cost_base_bindings_.push_back(b);
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
   * @brief Returns a vector of bindings to CostBase<MatrixType> for each
   * binding within the program.
   *
   * @return const std::vector<Binding<CostBase>>&
   */
  const std::vector<Binding<CostBase>>& GetCostBindings() const {
    return cost_base_bindings_;
  }

  /**
   * @brief Returns a vector of bindings to ConstraintBase<MatrixType> for each
   * binding within the program. Useful for iterating through all constraint's
   * base classes.
   *
   * @return const std::vector<Binding<ConstraintBase>>&
   */
  const std::vector<Binding<ConstraintBase>>& GetConstraintBindings() const {
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
        GetCurrentProgram().GetDecisionVariableIndices(var);
    return decision_variable_cache_(indices);
  }

  /**
   * @brief All constraints within the program
   *
   * @return std::vector<Binding<ConstraintBase>>&
   */
  std::vector<Binding<ConstraintBase>>& GetConstraints() {
    return constraints_;
  }

  /**
   * @brief All costs within the program
   *
   * @return std::vector<Binding<CostBase>>&
   */
  std::vector<Binding<CostBase>>& GetCosts() { return costs_; }

  template <typename T>
  void SetBindingInputs(const Binding<T>& binding) {
    VLOG(10) << "GetBindingInputs()";
    // Get binding data
    BindingInputData& data = GetBindingInputData(binding);
    // Variables
    int mapped_cnt = 0, manual_cnt = 0;
    for (int i = 0; i < binding.nx(); ++i) {
      const sym::VariableVector& xi = binding.x(i);
      VLOG(10) << "xi = " << xi;
      if (data.x_continuous[i]) {
        VLOG(10) << "continuous";
        // Provide mapped vector
        x.push_back(data.x_mapped_vecs[mapped_cnt++]);
      } else {
        VLOG(10) << "not continuous";
        VLOG(10) << data.x_manual_vecs.size();
        // Construct a vector for this input
        for (int j = 0; j < xi.size(); ++j) {
          data.x_manual_vecs[manual_cnt][j] =
              decision_variable_cache_[GetCurrentProgram()
                                           .GetDecisionVariableIndex(xi[j])];
        }
        x.push_back(data.x_manual_vecs[manual_cnt++]);
      }
    }
    // Parameters
    mapped_cnt = 0, manual_cnt = 0;
    for (int i = 0; i < binding.np(); ++i) {
      const sym::ParameterVector& pi = binding.p(i);
      VLOG(10) << "pi = " << pi;
      if (data.p_continuous[i]) {
        VLOG(10) << "continuous";
        // Provide mapped vector
        p.push_back(data.p_mapped_vecs[mapped_cnt++]);
      } else {
        VLOG(10) << "not continuous";
        VLOG(10) << data.p_manual_vecs.size();
        // Construct a vector for this input
        for (int j = 0; j < pi.size(); ++j) {
          data.p_manual_vecs[manual_cnt][j] =
              decision_variable_cache_[GetCurrentProgram().GetParameterIndex(
                  pi[j])];
        }
        p.push_back(data.p_manual_vecs[manual_cnt++]);
      }
    }
    VLOG(10) << "Finished";
  }

  void EvaluateCost(const Binding<CostType>& binding, const Eigen::VectorXd& x,
                    bool grd, bool hes, bool update_cache = true);

  void EvaluateCosts(const Eigen::VectorXd& x, bool grd, bool hes);

  // Evaluates the constraint and updates the cache for the gradients
  void EvaluateConstraint(const Binding<ConstraintType>& binding,
                          const int& constraint_idx, const Eigen::VectorXd& x,
                          bool jac, bool update_cache = true);

  void EvaluateConstraints(const Eigen::VectorXd& x, bool jac);

  void UpdateConstraintJacobian(const Binding<ConstraintType>& binding,
                                const int& constraint_idx);

  void UpdateLagrangianHessian(const Binding<CostType>& binding);

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

  // Primal solution of the program
  Eigen::VectorXd primal_solution_x_;

  // Vector of the constraint bindings for the program
  std::vector<Binding<ConstraintBase>> constraints_;
  // Vector of the cost bindings for the program
  std::vector<Binding<CostBase>> costs_;

  /**
   * @brief Binding data related to whether the inputs are continuous within
   * the optimisation vector as well as mappings and vectors for these inputs
   *
   */
  struct BindingInputData {
    BindingInputData()
        : x_continuous({}),
          x_mapped_vecs({}),
          x_manual_vecs({}),
          p_continuous({}),
          p_mapped_vecs({}),
          p_manual_vecs({}) {}
    // Flags to whether the variables x[i] are continuous within the
    // optimisation vector
    std::vector<bool> x_continuous;
    std::vector<Eigen::Map<const Eigen::VectorXd>> x_mapped_vecs;
    std::vector<Eigen::VectorXd> x_manual_vecs;

    std::vector<bool> p_continuous;
    std::vector<Eigen::Map<const Eigen::VectorXd>> p_mapped_vecs;
    std::vector<Eigen::VectorXd> p_manual_vecs;
  };

  /**
   * @brief Provides a reference to the input data associated with the binding
   *
   * @tparam T
   * @param binding
   * @return BindingInputData&
   */
  template <typename T>
  BindingInputData& GetBindingInputData(const Binding<T>& binding) {
    VLOG(10) << "GetBindingInputData()";

    auto idx = binding_idx_.find(binding.id());
    if (idx == binding_idx_.end()) {
      throw std::runtime_error("Binding is not included within program");
    }
    // Return the binding data given by the index
    VLOG(10) << "Found Binding Data for " << binding.Get().name()
             << " at Index " << idx->second;
    return data_[idx->second];
  }

 private:
  bool is_solved_ = false;
  // Reference to current program in solver
  Program& program_;
  // Index of the constraints within the constraint vector
  std::vector<int> constraint_idx_;

  // Provides data for each variable
  std::unordered_map<int, int> binding_idx_;
  std::vector<BindingInputData> data_ = {};
  std::vector<Binding<CostBase>> cost_base_bindings_ = {};
  std::vector<Binding<ConstraintBase>> constraint_base_bindings_ = {};

  /**
   * @brief Register a binding with the solver
   *
   * @tparam T
   * @param binding
   * @param manager
   * @param x
   */
  template <typename T>
  void RegisterBinding(const Binding<T>& binding) {
    VLOG(10) << "RegisterBinding()";
    BindingInputData data;
    // Assess if each input for the binding is continuous in the given
    // decision variable vector
    for (int i = 0; i < binding.nx(); ++i) {
      const sym::VariableVector& vi = binding.x(i);
      if (GetCurrentProgram().IsContinuousInDecisionVariableVector(vi)) {
        // Map vector slice from x to binding input
        data.x_mapped_vecs.push_back(Eigen::Map<const Eigen::VectorXd>(
            decision_variable_cache_.data() +
                GetCurrentProgram().GetDecisionVariableIndex(vi[0]),
            vi.size()));
        data.x_continuous.push_back(true);
      } else {
        // Create vector to manually place data in
        data.x_manual_vecs.push_back(Eigen::VectorXd::Zero(vi.size()));
        data.x_continuous.push_back(false);
      }
    }

    // Assess if each input for the binding is continuous in the given
    // parameter vector
    for (int i = 0; i < binding.np(); ++i) {
      const sym::ParameterVector& pi = binding.p(i);
      if (GetCurrentProgram().IsContinuousInParameterVector(pi)) {
        // Map vector slice from x to binding input
        data.p_mapped_vecs.push_back(Eigen::Map<const Eigen::VectorXd>(
            GetCurrentProgram().GetParameterVector().data() +
                GetCurrentProgram().GetParameterIndex(pi[0]),
            pi.size()));
        data.p_continuous.push_back(true);
      } else {
        // Create vector to manually place data in
        data.p_manual_vecs.push_back(Eigen::VectorXd::Zero(pi.size()));
        data.p_continuous.push_back(false);
      }
    }

    // Register data
    binding_idx_[binding.id()] = data_.size();
    data_.push_back(data);

    VLOG(10) << "Binding Data ID " << binding.id();
    VLOG(10) << "Binding Data at Index " << binding_idx_[binding.id()];
  }
};

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_BASE_H */
