#ifndef SOLVERS_SOLVER_H
#define SOLVERS_SOLVER_H

#include "damotion/solvers/base.h"

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
    constraint_jacobian_cache_ = Eigen::MatrixXd::Zero(
        program.NumberOfConstraints(), program.NumberOfDecisionVariables());
    lagrangian_hes_cache_ =
        Eigen::MatrixXd::Zero(program.NumberOfDecisionVariables(),
                              program.NumberOfDecisionVariables());
  }
  ~Solver() = default;

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
  Eigen::MatrixXd constraint_jacobian_cache_;
  Eigen::MatrixXd lagrangian_hes_cache_;

 private:
};

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_SOLVER_H */
