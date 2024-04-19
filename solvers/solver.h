#ifndef SOLVERS_SOLVER_H
#define SOLVERS_SOLVER_H

#include "solvers/base.h"

namespace damotion {
namespace optimisation {
namespace solvers {

/**
 * @brief Solver class for dense problems such that all matrices are dense
 * objects, ideal for small problems of low dimension and those run at high
 * rates such as for MPC
 *
 */
class Solver : public SolverBase<Eigen::MatrixXd> {
   public:
    Solver(Program& program) : SolverBase<Eigen::MatrixXd>(program) {
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
                                          bool is_block_x, bool is_block_y);

   private:
    Eigen::VectorXd constraint_cache_;
    Eigen::MatrixXd constraint_jacobian_cache_;
};


}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

#endif/* SOLVERS_SOLVER_H */
