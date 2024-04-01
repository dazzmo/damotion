#ifndef SOLVERS_SOLVER_H
#define SOLVERS_SOLVER_H

#include "solvers/program.h"

namespace damotion {
namespace optimisation {
namespace solvers {

class SolverBase {
   public:
    SolverBase(Program& prog);
    ~SolverBase() {}

    /**
     * @brief Returns the current program that is being solved. Allows for
     * constraints and bounds to be modified explicitly. If the internal
     * structure of the program is to be modified, please create a new instance
     * of a solver for it
     *
     * @return Program&
     */
    Program& GetCurrentProgram() { return prog_; }

    /**
     * @brief Updates the existing solver with an update program (e.g.
     * parameters or bounds have changed)
     *
     * @param program
     */
    void UpdateProgram(Program& program) { prog_ = program; }

    void EvaluateCost(Cost& c, const Eigen::VectorXd& x,
                      const std::vector<sym::VariableVector>& var,
                      const std::vector<bool>& continuous, bool grd, bool hes,
                      bool update_cache = true);

    void EvaluateCosts(const Eigen::VectorXd& x, bool grd, bool hes);

    // Evaluates the constraint and updates the cache for the gradients
    void EvaluateConstraint(Constraint& c, const int& constraint_idx,
                            const Eigen::VectorXd& x,
                            const std::vector<sym::VariableVector>& var,
                            const std::vector<bool>& continuous, bool jac,
                            bool update_cache = true);

    void EvaluateConstraints(const Eigen::VectorXd& x, bool jac);

    const Eigen::VectorXd& GetPrimalSolution() const {
        return primal_solution_x_;
    }

    std::vector<Binding<Constraint>>& GetConstraints() { return constraints_; }
    std::vector<Binding<Cost>>& GetCosts() { return costs_; }

   protected:
    // Solver caches
    Eigen::VectorXd decision_variable_cache_;
    Eigen::VectorXd dual_variable_cache_;

    double objective_cache_;
    Eigen::VectorXd objective_gradient_cache_;

    Eigen::VectorXd constraint_cache_;
    Eigen::MatrixXd constraint_jacobian_cache_;

    Eigen::MatrixXd lagrangian_hes_cache_;

    Eigen::VectorXd primal_solution_x_;

    // Vector of constraint bindings
    std::vector<Binding<Constraint>> constraints_;
    std::vector<Binding<Cost>> costs_;

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
    Program& prog_;

    // Calculate binding inputs for the problem
    void CalculateBindingInputs();

    std::unordered_map<Binding<Constraint>::Id, int> constraint_binding_idx;
    std::unordered_map<Binding<Cost>::Id, int> cost_binding_idx;

    std::vector<std::vector<bool>> constraint_binding_continuous_input_;
    std::vector<std::vector<bool>> cost_binding_continuous_input_;

    bool IsContiguousInDecisionVariableVector(const sym::VariableVector& var);
};

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_SOLVER_H */
