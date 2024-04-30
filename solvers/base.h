#ifndef SOLVERS_BASE_H
#define SOLVERS_BASE_H

#include <unordered_map>

#include "optimisation/program.h"

namespace damotion {
namespace optimisation {
namespace solvers {

template <typename MatrixType>
class SolverBase {
   public:
    typedef ProgramBase<MatrixType> ProgramType;
    typedef ConstraintBase<MatrixType> ConstraintType;
    typedef CostBase<MatrixType> CostType;

    SolverBase(ProgramType& prog) : prog_(prog) {
        int nx = prog_.NumberOfDecisionVariables();
        int nc = prog_.NumberOfConstraints();
        // Initialise vectors
        decision_variable_cache_ = Eigen::VectorXd::Zero(nx);
        primal_solution_x_ = Eigen::VectorXd::Zero(nx);
        objective_gradient_cache_ = Eigen::VectorXd::Zero(nx);
        constraint_cache_ = Eigen::VectorXd::Zero(nc);
        dual_variable_cache_ = Eigen::VectorXd::Zero(nc);

        CalculateBindingInputs();
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
    ProgramType& GetCurrentProgram() { return prog_; }

    /**
     * @brief Updates the existing solver with an update program (e.g.
     * parameters or bounds have changed)
     *
     * @param program
     */
    void UpdateProgram(ProgramType& program) { prog_ = program; }

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
     * @param data
     */
    void UpdateCostGradient(const Binding<CostType>& binding,
                            const BindingInputData& data) {
        for (int i = 0; i < binding.NumberOfVariables(); ++i) {
            if (data.continuous[i]) {
                // Block-insert
                objective_gradient_cache_.middleRows(
                    GetCurrentProgram().GetDecisionVariableIndex(var[0]),
                    var.size()) += binding.Get().Gradient(i);
            } else {
                // Manually insert into vector
                for (int j = 0; j < binding.GetVariable(i).size(); ++j) {
                    int idx =
                        GetCurrentProgram().GetDecisionVariableIndex(var[j]);
                    objective_gradient_cache_[idx] +=
                        binding.Get().Gradient(i)[j];
                }
            }
        }
    }

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
        return binding_input_data_[idx->second];
    }

    /**
     * @brief Create input vectors for the given binding from the optimisation
     * vector x.
     *
     * @tparam T
     * @param binding
     * @param data Data
     * @param x
     * @return std::vector<Eigen::Map<Eigen::VectorXd>>
     */
    template <typename T>
    void UpdateBindingInputData(Binding<T>& binding, const Eigen::VectorXd& x,
                                BindingInputData& data) {
        // Get the binding index
        int cnt = 0;
        for (int i = 0; i < binding.NumberOfVariables(); ++i) {
            if (data.continuous[i]) {
                // Set input to the start of the vector input
                data.inputs[i] = Eigen::Map<const Eigen::VectorXd>(
                    x.data() + GetCurrentProgram().GetDecisionVariableIndex(
                                   binding.GetVariable(i)[0]),
                    binding.GetVariable(i).size());
            } else {
                // Construct a vector for this input
                for (int ii = 0; ii < var[i]->size(); ++ii) {
                    data.vecs[cnt][ii] =
                        x[GetCurrentProgram().GetDecisionVariableIndex(
                            binding.GetVariable(i)[ii])];
                }
                data.inputs = data.vecs[cnt];
                cnt++;
            }
        }
    }

   protected:
    /**
     * @brief Input data for each binding, providing vectors extracted from the
     * optimisation vector of the program to provide input to the binding
     *
     */
    struct BindingInputData {
        // Inputs vectors for a binding
        std::vector<Eigen::Map<const Eigen::VectorXd>> inputs;
        // Manually created vectors for non-continuous data in the optimisation
        // vector
        std::vector<Eigen::VectorXd> vecs;
        // Whether the input vector for the binding is continuous in the
        // optimisation vector
        bool continuous;
    };

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

    template <typename T>
    const std::vector<bool>& ConstraintBindingContinuousInputCheck(
        const Binding<T>& binding) {
        return constraint_binding_continuous_input_
            [constraint_binding_idx_[binding.id()]];
    }

    template <typename T>
    const int& GetBindingIndex(const Binding<T>& binding) {
        // TODO - Make sure the binding exists
        return binding_idx_[binding.id()];
    }

    template <typename T>
    const std::vector<bool>& CostBindingContinuousInputCheck(
        const Binding<T>& binding) {
        return cost_binding_continuous_input_[cost_binding_idx_[binding.id()]];
    }

   private:
    ProgramType& prog_;

    bool is_solved_ = false;

    // Calculate binding inputs for the problem
    void CalculateBindingInputs() {
        ProgramType& program = GetCurrentProgram();
        // Compute binding input data for all constraints
        for (Binding<ConstraintType>& b : program.GetAllConstraintBindings()) {
            // Create binding data
            BindingInputData data;
            data.continuous.resize(b.NumberOfVariables());
            data.inputs.resize(b.NumberOfVariables());
            for (int i = 0; i < b.NumberOfVariables(); ++i) {
                if (IsContinuousInDecisionVariableVector(b.GetVariable(i))) {
                    data.continuous[i] = true;
                } else {
                    data.continuous[i] = false;
                    data.vecs.push_back(
                        Eigen::VectorXd::Zero(b.GetVariable(i).size()));
                }
            }
            // Add index to map
            binding_idx_[b.id()] = binding_input_data_.size();
            // Include continuous data
            binding_input_data_.push_back(data);
        }
        // Compute binding input data for all costs
        for (Binding<CostType>& b : program.GetAllCostBindings()) {
            // Create binding data
            BindingInputData data;
            data.continuous.resize(b.NumberOfVariables());
            data.inputs.resize(b.NumberOfVariables());
            for (int i = 0; i < b.NumberOfVariables(); ++i) {
                if (IsContinuousInDecisionVariableVector(b.GetVariable(i))) {
                    data.continuous[i] = true;
                } else {
                    data.continuous[i] = false;
                    data.vecs.push_back(
                        Eigen::VectorXd::Zero(b.GetVariable(i).size()));
                }
            }
            // Add index to map
            binding_idx_[b.id()] = binding_input_data_.size();
            // Include continuous data
            binding_input_data_.push_back(data);
        }
    }

    /**
     * @brief Determines whether a vector of variables var is continuous within
     * the optimisation vector of the program.
     *
     * @param var
     * @return true
     * @return false
     */
    bool IsContinuousInDecisionVariableVector(const sym::VariableVector& var) {
        VLOG(10) << "IsContinuousInDecisionVariableVector(), checking " << var;
        ProgramType& program = GetCurrentProgram();
        // Determine the index of the first element within var
        int idx = program.GetDecisionVariableIndex(var[0]);
        // Move through optimisation vector and see if each entry follows one
        // after the other
        for (int i = 1; i < var.size(); ++i) {
            // If not, return false
            int idx_next = program.GetDecisionVariableIndex(var[i]);

            if (idx_next - idx != 1) {
                VLOG(10) << "false";
                return false;
            }
            idx = idx_next;
        }
        // Return true if all together in the vector
        VLOG(10) << "true";
        return true;
    }

    // Indices for each binding data
    std::unordered_map<int, int> binding_idx_;

    // Vector of binding input data structs
    std::vector<BindingInputData> binding_input_data_;

    // Index of the constraints within the constraint vector
    std::vector<int> constraint_idx_;
};

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_BASE_H */
