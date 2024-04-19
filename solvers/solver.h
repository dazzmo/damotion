#ifndef SOLVERS_SOLVER_H
#define SOLVERS_SOLVER_H

#include "solvers/program.h"

namespace damotion {
namespace optimisation {
namespace solvers {

template <typename MatrixType>
class SolverBase {
   public:
    typedef ProgramBase<MatrixType> ProgramType;
    typedef ConstraintBase<MatrixType> ConstraintType;
    typedef CostBase<MatrixType> CostType;

    SolverBase(ProgramType& prog) : prog_(prog), sparse_(sparse) {
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
     * @param vec
     * @param vals
     * @param var
     * @param is_block
     */
    void UpdateVectorAtVariableLocations(Eigen::VectorXd& vec,
                                         const Eigen::VectorXd& val,
                                         const sym::VariableVector& var,
                                         bool is_block) {
        if (is_block) {
            vec.middleRows(GetCurrentProgram().GetDecisionVariableIndex(var[0]),
                           var.size()) += block;
        } else {
            // For each variable, update the location in the Jacobian
            for (int j = 0; j < var.size(); ++j) {
                int idx = GetCurrentProgram().GetDecisionVariableIndex(var[j]);
                vec[idx] += block[j];
            }
        }
    }

   protected:
    // Solver caches
    Eigen::VectorXd decision_variable_cache_;
    Eigen::VectorXd dual_variable_cache_;

    double objective_cache_;
    Eigen::VectorXd objective_gradient_cache_;

    Eigen::MatrixXd lagrangian_hes_cache_;

    Eigen::VectorXd primal_solution_x_;

    // Vector of constraint bindings
    std::vector<Binding<ConstraintType>> constraints_;
    std::vector<Binding<CostType>> costs_;

    template <typename T>
    const std::vector<bool>& ConstraintBindingContinuousInputCheck(
        const Binding<T>& binding) {
        return constraint_binding_continuous_input_
            [constraint_binding_idx[binding.id()]];
    }

    template <typename T>
    const std::vector<bool>& CostBindingContinuousInputCheck(
        const Binding<T>& binding) {
        return cost_binding_continuous_input_[cost_binding_idx[binding.id()]];
    }

   private:
    ProgramType& prog_;

    bool is_solved_ = false;

    // Calculate binding inputs for the problem
    void CalculateBindingInputs() {
        ProgramType& program = GetCurrentProgram();
        // Pre-compute variable look ups for speed
        for (Binding<ConstraintType>& b : program.GetAllConstraints()) {
            // For each input, assess if memory is continuous (allow for
            // optimisation of input)
            std::vector<bool> continuous(b.NumberOfVariables());
            for (int i = 0; i < b.NumberOfVariables(); ++i) {
                continuous[i] =
                    IsContinuousInDecisionVariableVector(b.GetVariable(i));
            }
            // Add index to map
            constraint_binding_idx[b.id()] =
                constraint_binding_continuous_input_.size();
            // Include continuous data
            constraint_binding_continuous_input_.push_back(continuous);
        }

        // Perform the same for the costs
        for (Binding<CostType>& b : program.GetAllCostBindings()) {
            // For each input, assess if memory is continuous (allow for
            // optimisation of input)
            std::vector<bool> continuous(b.NumberOfVariables());
            for (int i = 0; i < b.NumberOfVariables(); ++i) {
                continuous[i] =
                    IsContinuousInDecisionVariableVector(b.GetVariable(i));
            }
            // Add index to map
            cost_binding_idx[b.id()] = cost_binding_continuous_input_.size();
            // Include continuous data
            cost_binding_continuous_input_.push_back(is_continuous);
        }
    }

    bool IsContinuousInDecisionVariableVector(const sym::VariableVector& var) {
        ProgramType& program = GetCurrentProgram();
        int idx = program.GetDecisionVariableIndex(var[0]);
        // Move through optimisation vector and see if each entry follows one
        // after the other
        for (int i = 1; i < var.size(); ++i) {
            // If not, return false
            if (program.GetDecisionVariableIndex(var[i]) - idx != 1) {
                return false;
            }
            idx = program.GetDecisionVariableIndex(var[i]);
        }
        // Return true if all together in the vector
        return true;
    }

    std::unordered_map<Binding<ConstraintType>::Id, int> constraint_binding_idx;
    std::unordered_map<Binding<CostType>::Id, int> cost_binding_idx;

    // Index of the constraints within the constraint vector
    std::vector<int> constraint_idx_;

    std::vector<std::vector<bool>> constraint_binding_continuous_input_;
    std::vector<std::vector<bool>> cost_binding_continuous_input_;
};

/**
 * @brief Solver class for dense problems such that all matrices are dense
 * objects, ideal for small problems of low dimension and those run at high
 * rates such as for MPC
 *
 */
class Solver : public SolverBase<Eigen::MatrixXd> {
   public:
    Solver(const Program& program) : SolverBase<Eigen::MatrixXd>(program) {
        // Initialise dense constraint Jacobian and Lagrangian Hessian
    }
    ~Solver() = default;

    void EvaluateCost(Binding<CostType>& binding, const Eigen::VectorXd& x,
                      bool grd, bool hes, bool update_cache = true);

    void EvaluateCosts(const Eigen::VectorXd& x, bool grd, bool hes);

    // Evaluates the constraint and updates the cache for the gradients
    void EvaluateConstraint(Binding<ConstraintType>& binding,
                            const int& constraint_idx, const Eigen::VectorXd& x,
                            bool jac, bool update_cache = true);

    void EvaluateConstraints(const Eigen::VectorXd& x, bool jac);

    void UpdateJacobianAtVariableLocations(Eigen::MatrixXd& jac, int row_idx,
                                           const Eigen::MatrixXd& block,
                                           const sym::VariableVector& var,
                                           bool is_block);

    void UpdateHessianAtVariableLocations(Eigen::MatrixXd& hes,
                                          const Eigen::MatrixXd& block,
                                          const sym::VariableVector& var_x,
                                          const sym::VariableVector& var_y,
                                          bool is_block_x,
                                          bool is_block_y);

   private:
    Eigen::VectorXd constraint_cache_;
    Eigen::MatrixXd constraint_jacobian_cache_;
};

class SparseSolver : public SolverBase<Eigen::SparseMatrix<double>> {
   public:
    SparseSolver(SparseProgram& prog, bool sparse = false);
    ~SparseSolver() {}

    void EvaluateCost(Binding<CostType>& binding, const Eigen::VectorXd& x,
                      bool grd, bool hes, bool update_cache = true);

    void EvaluateCosts(const Eigen::VectorXd& x, bool grd, bool hes);

    // Evaluates the constraint and updates the cache for the gradients
    void EvaluateConstraint(Constraint& c, const int& constraint_idx,
                            const Eigen::VectorXd& x,
                            const std::vector<sym::VariableVector>& var,
                            const sym::ParameterVector& par,
                            const std::vector<bool>& continuous, bool jac,
                            bool update_cache = true);

    void EvaluateConstraints(const Eigen::VectorXd& x, bool jac);

   protected:
    Eigen::SparseMatrix<double> constraint_jacobian_cache_;
    Eigen::SparseMatrix<double> lagrangian_hes_cache_;

    // Vector of constraint bindings
    std::vector<Binding<SparseConstraint>> constraints_;
    std::vector<Binding<SparseCost>> costs_;

   private:
    // Calculate binding inputs for the problem
    void CalculateBindingInputs();

    std::unordered_map<Binding<SparseConstraint>::Id, int>
        constraint_binding_idx;
    std::unordered_map<Binding<SparseCost>::Id, int> cost_binding_idx;

    void ConstructSparseConstraintJacobian();
};

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_SOLVER_H */
