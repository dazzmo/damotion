#ifndef SOLVERS_SPARSE_H
#define SOLVERS_SPARSE_H

#include "solvers/base.h"

namespace damotion {
namespace optimisation {
namespace solvers {

class SparseSolver : public SolverBase<Eigen::SparseMatrix<double>> {
   public:
    SparseSolver(SparseProgram& program);
    ~SparseSolver() {}

    void EvaluateCost(Binding<CostType>& binding, const Eigen::VectorXd& x,
                      bool grd, bool hes, bool update_cache = true);

    void EvaluateCosts(const Eigen::VectorXd& x, bool grd, bool hes) {}

    // Evaluates the constraint and updates the cache for the gradients
    void EvaluateConstraint(const Binding<ConstraintType>& binding,
                            const int& constraint_idx, const Eigen::VectorXd& x,
                            bool jac, bool update_cache = true);

    void EvaluateConstraints(const Eigen::VectorXd& x, bool jac) {}

    void ConstructSparseConstraintJacobian();

    const Eigen::SparseMatrix<double>& GetSparseConstraintJacobian() const {
        return constraint_jacobian_cache_;
    }

   protected:
    Eigen::SparseMatrix<double> constraint_jacobian_cache_;
    Eigen::SparseMatrix<double> lagrangian_hes_cache_;

    // Vector of constraint bindings
    std::vector<Binding<ConstraintType>> constraints_;
    std::vector<Binding<CostType>> costs_;

   private:
    std::unordered_map<Binding<ConstraintType>::Id, int> constraint_binding_idx;
    std::unordered_map<Binding<CostType>::Id, int> cost_binding_idx;

    // Index of each sparse Jacobian's data entries in the constraint Jacobian
    std::unordered_map<Binding<ConstraintType>::Id,
                       std::vector<std::vector<int>>>
        jacobian_data_map_;
};
}  // namespace solvers
}  // namespace optimisation

}  // namespace damotion

#endif /* SOLVERS_SPARSE_H */
