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
                GetCurrentProgram().GetDecisionVariableIndex(var[0]),
                var.size());
        } else {
            Eigen::VectorXd vec(var.size());
            for (int i = 0; i < var.size(); ++i) {
                vec[i] =
                    primal_solution_x_[GetCurrentProgram()
                                           .GetDecisionVariableIndex(var[i])];
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
        ProgramType& program = GetCurrentProgram();
        BindingInputData& data = GetBindingInputData(binding);
        int g_idx = 0;
        for (int i = 0; i < binding.nx(); ++i) {
            const sym::VariableVector& xi = binding.x(i);
            // Get slice of the gradient vector
            Eigen::Ref<const Eigen::VectorXd> gi =
                binding.Get().Gradient().middleRows(g_idx, xi.size());
            if (data.continuous[i]) {
                // Block-insert
                objective_gradient_cache_.middleRows(
                    program.GetDecisionVariableIndex(xi[0]), xi.size()) += gi;
            } else {
                // Manually insert into vector
                for (int j = 0; j < xi.size(); ++j) {
                    int idx = program.GetDecisionVariableIndex(xi[j]);
                    objective_gradient_cache_[idx] += gi[j];
                }
            }
            g_idx += xi.size();
        }
    }

    template <typename T>
    void GetBindingInputs(const Binding<T>& binding,
                          common::InputRefVector& in) {
        // Get binding data
        BindingInputData& data = GetBindingInputData(binding);
        // Get the binding index
        int cnt = 0;
        int mapped_cnt = 0, manual_cnt = 0;
        for (int i = 0; i < binding.nx(); ++i) {
            const sym::VariableVector& vi = binding.x(i);
            if (data.continuous[i]) {
                // Provide mapped vector
                in.push_back(data.mapped_vecs[mapped_cnt++]);
            } else {
                // Construct a vector for this input
                for (int j = 0; j < vi.size(); ++j) {
                    data.manual_vecs[cnt][j] = decision_variable_cache_
                        [GetCurrentProgram().GetDecisionVariableIndex(vi[j])];
                }
                in.push_back(data.mapped_vecs[manual_cnt++]);
            }
        }
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
        std::vector<bool> continuous;
        std::vector<Eigen::Map<const Eigen::VectorXd>> mapped_vecs;
        std::vector<Eigen::VectorXd> manual_vecs;
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
        auto idx = binding_idx_.find(binding.id());
        if (idx == binding_idx_.end()) {
            throw std::runtime_error("Binding is not included within program");
        }
        // Return the binding data given by the index
        return data_[idx->second];
    }

   private:
    bool is_solved_ = false;
    // Reference to current program in solver
    ProgramType& program_;
    // Index of the constraints within the constraint vector
    std::vector<int> constraint_idx_;

    // Provides data for each variable
    std::unordered_map<int, int> binding_idx_;
    std::vector<BindingInputData> data_;

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
        BindingInputData data;
        // Assess if each input for the binding is continuous in the given
        // decision variable vector
        for (int i = 0; i < binding.nx(); ++i) {
            const sym::VariableVector& vi = binding.x(i);
            if (GetCurrentProgram().IsContinuousInDecisionVariableVector(vi)) {
                // Map vector slice from x to binding input
                data.mapped_vecs.push_back(Eigen::Map<const Eigen::VectorXd>(
                    decision_variable_cache_.data() +
                        GetCurrentProgram().GetDecisionVariableIndex(vi[0]),
                    vi.size()));
                data.continuous.push_back(true);
            } else {
                // Create vector to manually place data in
                data.manual_vecs.push_back(Eigen::VectorXd::Zero(vi.size()));
                data.continuous.push_back(false);
            }
        }
        // Register data
        binding_idx_[binding.id()] = data_.size();
        data_.push_back(data);
    }

    
};

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_BASE_H */
