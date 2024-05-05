#ifndef SOLVERS_BASE_H
#define SOLVERS_BASE_H

#include <unordered_map>

#include "damotion/optimisation/program.h"

namespace damotion {
namespace optimisation {
namespace solvers {

template <typename MatrixType>
class SolverBase {
 public:
  typedef ProgramBase<MatrixType> ProgramType;
  typedef ConstraintBase<MatrixType> ConstraintType;
  typedef CostBase<MatrixType> CostType;

  SolverBase(ProgramType& program) : program_(program) {
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
    }
    for (auto& b : program.GetAllCostBindings()) {
      RegisterBinding(b);
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
  ProgramType& GetCurrentProgram() { return program_; }

  /**
   * @brief Updates the existing solver with an update program (e.g.
   * parameters or bounds have changed)
   *
   * @param program
   */
  void UpdateProgram(ProgramType& program) { program_ = program; }

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
   * @brief Returns a vector of the current values of the variables provided
   * in var. If variables are all together in the vector, we can speed this
   * process up by setting is_continuous to true.
   *
   * @param var
   * @param is_continuous
   * @return Eigen::VectorXd
   */
  Eigen::VectorXd GetVariableValues(const sym::VariableVector& var,
                                    bool is_continuous = false) {
    if (is_continuous) {
      return primal_solution_x_.middleRows(
          GetCurrentProgram().GetDecisionVariableIndex(var[0]), var.size());
    } else {
      Eigen::VectorXd vec(var.size());
      for (int i = 0; i < var.size(); ++i) {
        vec[i] =
            primal_solution_x_[GetCurrentProgram().GetDecisionVariableIndex(
                var[i])];
      }
      return vec;
    }
  }

  /**
   * @brief All constraints within the program
   *
   * @return std::vector<Binding<ConstraintType>>&
   */
  std::vector<Binding<ConstraintType>>& GetConstraints() {
    return constraints_;
  }

  /**
   * @brief All costs within the program
   *
   * @return std::vector<Binding<CostType>>&
   */
  std::vector<Binding<CostType>>& GetCosts() { return costs_; }

  /**
   * @brief Updates the variables var within the vector vec with values val.
   * If the variables are within a block, indicating this with is_block can
   * enable efficient block-insertion.
   *
   * @param binding
   */
  void UpdateCostGradient(const Binding<CostType>& binding) {
    VLOG(10) << "UpdateCostGradient()";
    BindingInputData& data = GetBindingInputData(binding);
    int g_idx = 0;
    for (int i = 0; i < binding.nx(); ++i) {
      const sym::VariableVector& xi = binding.x(i);
      // Get slice of the gradient vector
      Eigen::Ref<const Eigen::VectorXd> gi =
          binding.Get().Gradient().middleRows(g_idx, xi.size());
      // Insert into the gradient cache
      InsertVectorAtVariableLocations(objective_gradient_cache_, gi, xi,
                                      data.x_continuous[i]);
      // Increase binding gradient index
      g_idx += xi.size();
    }
  }

  template <typename T>
  void GetBindingInputs(const Binding<T>& binding, common::InputRefVector& x,
                        common::InputRefVector& p) {
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
  Eigen::VectorXd constraint_cache_;

  // Primal solution of the program
  Eigen::VectorXd primal_solution_x_;

  // Vector of the constraint bindings for the program
  std::vector<Binding<ConstraintType>> constraints_;
  // Vector of the cost bindings for the program
  std::vector<Binding<CostType>> costs_;

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

  /**
   * @brief Places the vector vec within res specified at the variable
   * locations of var
   *
   * @param res
   * @param vec
   * @param var
   * @param is_continuous
   */
  void InsertVectorAtVariableLocations(Eigen::VectorXd& res,
                                       const Eigen::VectorXd& vec,
                                       const sym::VariableVector& var,
                                       bool is_continuous = false) {
    VLOG(10) << "InsertVectorAtVariableLocations()";
    if (is_continuous) {
      // Block-insert
      res.middleRows(program_.GetDecisionVariableIndex(var[0]), var.size()) +=
          vec;
    } else {
      // Manually insert into vector
      for (int j = 0; j < var.size(); ++j) {
        int idx = program_.GetDecisionVariableIndex(var[j]);
        res[idx] += vec[j];
      }
    }
  }

  void InsertJacobianAtVariableLocations(Eigen::MatrixXd& res,
                                         const Eigen::MatrixXd& jac,
                                         const sym::VariableVector& var,
                                         const int& constraint_idx,
                                         bool is_continuous = false) {
    VLOG(8) << "InsertJacobianAtVariableLocations()";
    VLOG(10) << "res\n" << res;
    VLOG(10) << "jac\n" << jac;
    VLOG(10) << "var\n" << var;
    VLOG(10) << "constraint index " << constraint_idx;
    Eigen::Ref<Eigen::MatrixXd> J = res.middleRows(constraint_idx, jac.rows());
    if (is_continuous) {
      J.middleCols(GetCurrentProgram().GetDecisionVariableIndex(var[0]),
                   var.size()) += jac;
    } else {
      VLOG(10) << "Manual Insertion";
      // For each variable, update the location in the Jacobian
      for (int i = 0; i < var.size(); ++i) {
        int idx = GetCurrentProgram().GetDecisionVariableIndex(var[i]);
        VLOG(10) << "i = " << i << " Index = " << idx;
        J.col(idx) += jac.col(i);
      }
    }
  }

  /**
   * @brief Inserts a Hessian matrix for the variables var_x and var_y into the
   * Hessian matrix res where var_x and var_y are sub-vectors of the Hessian.
   *
   * @param res
   * @param mat
   * @param var_x
   * @param var_y
   * @param is_continuous_x
   * @param is_continuous_y
   */
  void InsertHessianAtVariableLocations(Eigen::MatrixXd& res,
                                        const Eigen::MatrixXd& mat,
                                        const sym::VariableVector& var_x,
                                        const sym::VariableVector& var_y,
                                        bool is_continuous_x = false,
                                        bool is_continuous_y = false) {
    VLOG(10) << "InsertHessianAtVariableLocations()";
    // For each variable combination
    if (is_continuous_x && is_continuous_y) {
      int x_idx = program_.GetDecisionVariableIndex(var_x[0]);
      int y_idx = program_.GetDecisionVariableIndex(var_y[0]);
      // Create lower triangular Hessian
      if (x_idx > y_idx) {
        res.block(x_idx, y_idx, var_x.size(), var_y.size()) += mat;
      } else {
        res.block(y_idx, x_idx, var_y.size(), var_x.size()) += mat.transpose();
      }

    } else {
      // For each variable pair, populate the Hessian
      for (int i = 0; i < var_x.size(); ++i) {
        int x_idx = program_.GetDecisionVariableIndex(var_x[i]);
        for (int j = 0; j < var_y.size(); ++j) {
          int y_idx = program_.GetDecisionVariableIndex(var_y[j]);
          // Create lower triangular matrix
          if (x_idx > y_idx) {
            res(x_idx, y_idx) += mat(i, j);
          } else {
            res(y_idx, x_idx) += mat(i, j);
          }
        }
      }
    }
  }

 private:
  bool is_solved_ = false;
  // Reference to current program in solver
  ProgramType& program_;
  // Index of the constraints within the constraint vector
  std::vector<int> constraint_idx_;

  // Provides data for each variable
  std::unordered_map<int, int> binding_idx_;
  std::vector<BindingInputData> data_ = {};

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
