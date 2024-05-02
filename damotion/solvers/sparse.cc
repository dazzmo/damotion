#include "damotion/solvers/sparse.h"

namespace damotion {
namespace optimisation {
namespace solvers {

SparseSolver::SparseSolver(SparseProgram& program)
    : SolverBase<Eigen::SparseMatrix<double>>(program) {
    // Initialise sparse constraint Jacobian and Lagrangian Hessian
    ConstructSparseConstraintJacobian();
}

void SparseSolver::ConstructSparseConstraintJacobian() {
    SparseProgram& program = GetCurrentProgram();
    // Create default data structure to hold binding id, jacobian idx and data
    // idx
    struct JacobianIndexData {
        int binding_id;
        int variable_idx;
    };

    int nx = program.NumberOfDecisionVariables();
    int nc = program.NumberOfConstraints();
    // Create sparse Jacobian
    Eigen::SparseMatrix<std::shared_ptr<JacobianIndexData>> J;
    constraint_jacobian_cache_.resize(nc, nx);
    J.resize(nc, nx);
    // Make sure there's no existing data in the sparse matrix
    J.setZero();
    J.data().squeeze();

    constraint_jacobian_cache_.setZero();
    constraint_jacobian_cache_.data().squeeze();

    int idx = 0;
    for (Binding<ConstraintType>& b : program.GetAllConstraintBindings()) {
        // Create data within the map
        // Add constraint to jacobian
        // Get sparse Jacobian
        const Eigen::SparseMatrix<double>& Ji = b.Get().Jacobian();
        const sym::VariableVector& v = b.GetConcatenatedVariableVector();
        std::vector<int> indices(Ji.nonZeros());
        int cnt = 0;
        // Loop through non-zero entries
        for (int k = 0; k < Ji.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Ji, k); it;
                 ++it) {
                std::shared_ptr<JacobianIndexData> data;
                data->binding_id = b.id();
                data->variable_idx = cnt;
                // Constraint index
                int c_idx = idx + it.row();
                // Index for variable
                int x_idx = program.GetDecisionVariableIndex(v[it.col()]);
                // Set element in the full Jacobian to the information for
                // the Jacobian block provided
                J.coeffRef(c_idx, x_idx) = data;
                // Set structure for Jacobian cache
                constraint_jacobian_cache_.coeffRef(c_idx, x_idx) = 0.0;
                // Increase data array counter
                cnt++;
            }
        }

        // Increase constraint index
        idx += b.Get().Dimension();
        jacobian_data_map_[b.id()] = indices;
    }

    // Compress Jacobian
    J.makeCompressed();
    constraint_jacobian_cache_.makeCompressed();

    // With created Jacobian, extract each binding's Jacobian data entries in
    // the data vector in CCS
    for (int i = 0; i < J.nonZeros(); ++i) {
        std::shared_ptr<JacobianIndexData> data = J.valuePtr()[i];
        jacobian_data_map_[data->binding_id][data->variable_idx] = i;
    }
}

void SparseSolver::ConstructSparseLagrangianHessian(bool with_constraints) {
    SparseProgram& program = GetCurrentProgram();
    // Create default data structure to hold binding id, jacobian idx and data
    // idx
    struct HessianIndexData {
        int binding_id;
        int variable_idx;
    };
    int nx = program.NumberOfDecisionVariables();
    // Create sparse Jacobian
    Eigen::SparseMatrix<std::shared_ptr<HessianIndexData>> H;
    lagrangian_hes_cache_.resize(nx, nx);
    H.resize(nx, nx);
    // Make sure there's no existing data in the sparse matrix
    H.setZero();
    H.data().squeeze();

    lagrangian_hes_cache_.setZero();
    lagrangian_hes_cache_.data().squeeze();

    for (Binding<CostType>& b : program.GetAllCostBindings()) {
        // Create data within the map
        const Eigen::SparseMatrix<double>& Hi = b.Get().Hessian();
        std::vector<int> indices(Hi.nonZeros());
        const sym::VariableVector& v = b.GetConcatenatedVariableVector();

        // Get sparse Jacobian
        int cnt = 0;
        // Loop through non-zero entries
        for (int k = 0; k < Hi.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Hi, k); it;
                 ++it) {
                std::shared_ptr<HessianIndexData> data;
                data->binding_id = b.id();
                data->variable_idx = cnt;
                // Index for variable
                int x_idx = program.GetDecisionVariableIndex(v[it.row()]);
                int y_idx = program.GetDecisionVariableIndex(v[it.col()]);
                // Set element in the full Jacobian to the information
                // for the Jacobian block provided
                H.coeffRef(x_idx, y_idx) = data;
                // Set structure for Jacobian cache
                lagrangian_hes_cache_.coeffRef(x_idx, y_idx) = 0.0;
                // Increase data array counter
                cnt++;
            }
        }

        // Add to map
        lagrangian_hes_data_map_[b.id()] = indices;
    }

    // TODO - Perform the same for the constraints

    // Compress Jacobian
    H.makeCompressed();
    constraint_jacobian_cache_.makeCompressed();

    // With created Jacobian, extract each binding's Jacobian data entries in
    // the data vector in CCS
    for (int i = 0; i < H.nonZeros(); ++i) {
        std::shared_ptr<HessianIndexData> data = H.valuePtr()[i];
        // Add to the map
        lagrangian_hes_data_map_[data->binding_id][data->variable_idx] = i;
    }
}

void SparseSolver::EvaluateCost(Binding<CostType>& binding,
                                const Eigen::VectorXd& x, bool grd, bool hes,
                                bool update_cache) {
    common::InputRefVector x_in = {}, p_in = {};
    GetBindingInputs(binding, x_in);

    // Set parameters
    for (int i = 0; i < binding.np(); ++i) {
        p_in.push_back(GetCurrentProgram().GetParameterValues(binding.p(i)));
    }

    const CostType& cost = binding.Get();
    // Evaluate the constraint
    cost.eval(x_in, p_in, grd);
    // if(hes) constraint.eval_hessian(x_in, p_in, l_in)

    if (update_cache == false) return;

    objective_cache_ += cost.Objective();
    if (grd) UpdateCostGradient(binding);
    if (hes) UpdateLagrangianHessian(binding);
}

void SparseSolver::EvaluateConstraint(const Binding<ConstraintType>& binding,
                                      const int& constraint_idx,
                                      const Eigen::VectorXd& x, bool jac,
                                      bool update_cache) {
    common::InputRefVector x_in = {}, p_in = {};
    GetBindingInputs(binding, x_in);

    // Set parameters
    for (int i = 0; i < binding.np(); ++i) {
        p_in.push_back(GetCurrentProgram().GetParameterValues(binding.p(i)));
    }

    const ConstraintType& constraint = binding.Get();
    // Evaluate the constraint
    constraint.eval(x_in, p_in, jac);
    // if(hes) constraint.eval_hessian(x_in, p_in, l_in)

    // Update the caches if required, otherwise break early
    if (update_cache == false) return;

    constraint_cache_.middleRows(constraint_idx, constraint.Dimension()) =
        constraint.Vector();
    VLOG(10) << "constraint_cache = " << constraint_cache_;

    UpdateConstraintJacobian(binding);
}

void SparseSolver::UpdateConstraintJacobian(
    const Binding<ConstraintType>& binding) {
    // For each Jacobian, update the data within the constraint Jacobian
    for (int k = 0; k < binding.Get().Jacobian().nonZeros(); ++k) {
        int idx = jacobian_data_map_[binding.id()][k];
        constraint_jacobian_cache_.valuePtr()[idx] =
            binding.Get().Jacobian().valuePtr()[k];
    }
}

void SparseSolver::UpdateLagrangianHessian(const Binding<CostType>& binding) {
    // For each Jacobian, update the data within the constraint Jacobian
    for (int k = 0; k < binding.Get().Hessian().nonZeros(); ++k) {
        int idx = lagrangian_hes_data_map_[binding.id()][k];
        lagrangian_hes_cache_.valuePtr()[idx] +=
            binding.Get().Hessian().valuePtr()[k];
    }
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion