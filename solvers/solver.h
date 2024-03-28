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

    void EvaluateCost(Binding<Cost>& b, const Eigen::VectorXd& x, bool grd,
                      bool hes);
    void EvaluateCosts(const Eigen::VectorXd& x, bool grad, bool hes);

    // Evaluates the constraint and updates the cache for the gradients
    void EvaluateConstraint(Binding<Constraint>& b, const int& constraint_idx,
                            const Eigen::VectorXd& x, bool jac);

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

   private:
    Program& prog_;
};

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_SOLVER_H */
