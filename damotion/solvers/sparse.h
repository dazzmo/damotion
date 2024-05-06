#ifndef SOLVERS_SPARSE_H
#define SOLVERS_SPARSE_H

#include "damotion/solvers/base.h"

namespace damotion {
namespace optimisation {
namespace solvers {

class SparseSolver : public SolverBase<Eigen::SparseMatrix<double>> {
 public:
  SparseSolver(SparseProgram& program);
  ~SparseSolver() {}

  void EvaluateCost(const Binding<CostType>& binding, const Eigen::VectorXd& x,
                    bool grd, bool hes, bool update_cache = true);

  void EvaluateCosts(const Eigen::VectorXd& x, bool grd, bool hes);

  // Evaluates the constraint and updates the cache for the gradients
  void EvaluateConstraint(const Binding<ConstraintType>& binding,
                          const int& constraint_idx, const Eigen::VectorXd& x,
                          bool jac, bool hes, bool update_cache = true);

  void EvaluateConstraints(const Eigen::VectorXd& x, bool jac, bool hes);

  void ConstructSparseConstraintJacobian();
  void ConstructSparseLagrangianHessian(bool with_constraints = true);

  const Eigen::SparseMatrix<double>& GetSparseConstraintJacobian() const {
    return constraint_jacobian_cache_;
  }

  const Eigen::SparseMatrix<double>& GetSparseLagrangianHessian() const {
    return lagrangian_hes_cache_;
  }

  void UpdateConstraintJacobian(const Binding<ConstraintType>& binding);

  void UpdateLagrangianHessian(const Binding<CostType>& binding);
  void UpdateLagrangianHessian(const Binding<ConstraintType>& binding);

 protected:
  Eigen::SparseMatrix<double> constraint_jacobian_cache_;
  Eigen::SparseMatrix<double> lagrangian_hes_cache_;

  // Vector of constraint bindings
  std::vector<Binding<ConstraintType>> constraints_;
  std::vector<Binding<CostType>> costs_;

 private:
  // Index of each sparse Jacobian's data entries in the constraint Jacobian
  std::unordered_map<Binding<ConstraintType>::Id, std::vector<int>>
      jacobian_data_map_;
  std::unordered_map<Binding<ConstraintType>::Id, std::vector<int>>
      lagrangian_hes_data_map_;
};
}  // namespace solvers
}  // namespace optimisation

}  // namespace damotion

#endif /* SOLVERS_SPARSE_H */
