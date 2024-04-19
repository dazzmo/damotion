#ifndef SOLVERS_SPARSE_H
#define SOLVERS_SPARSE_H

#include "solvers/base.h"

namespace damotion {
namespace optimisation {
namespace solvers {

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
    std::unordered_map<Binding<SparseConstraint>::Id, int>
        constraint_binding_idx;
    std::unordered_map<Binding<SparseCost>::Id, int> cost_binding_idx;

    // Index of each sparse Jacobian's data entries in the constraint Jacobian
    std::unordered_map<Binding<SparseConstraint>::Id,
                       std::vector<std::vector<int>>>
        jacobian_data_map_;

    void ConstructSparseConstraintJacobian();
};
}  // namespace solvers
}  // namespace optimisation

}  // namespace damotion

#endif /* SOLVERS_SPARSE_H */
