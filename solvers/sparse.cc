#include "solvers/sparse.h"

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
        int jacobian_idx;
        int variable_idx;
    };

    typedef Eigen::Vector3i binding_index_data_t;
    int nx = program.NumberOfDecisionVariables();
    int nc = program.NumberOfConstraints();
    // Create sparse Jacobian
    Eigen::SparseMatrix<std::shared_ptr<binding_index_data_t>> J;
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
        std::vector<std::vector<int>> indices(b.NumberOfVariables());
        // Add constraint to jacobian
        for (int i = 0; i < b.GetVariables().size(); ++i) {
            // Get sparse Jacobian
            const Eigen::SparseMatrix<double>& Ji = b.Get().Jacobian(i);
            int cnt = 0;
            // Loop through non-zero entries
            for (int k = 0; k < Ji.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(Ji, k); it;
                     ++it) {
                    int id = b.id();
                    // Get location of the non-zero entry
                    int jac_idx = i;
                    // Constraint index
                    int c_idx = idx + it.row();
                    // Index for variable
                    int x_idx = program.GetDecisionVariableIndex(
                        b.GetVariable(i)[it.col()]);
                    // Set element in the full Jacobian to the information for
                    // the Jacobian block provided
                    J.coeffRef(c_idx, x_idx) =
                        std::make_shared<binding_index_data_t>(id, jac_idx,
                                                               cnt);
                    // Set structure for Jacobian cache
                    constraint_jacobian_cache_.coeffRef(c_idx, x_idx) = 0.0;
                    // Increase data array counter
                    cnt++;
                }
            }

            indices[i].resize(Ji.nonZeros());
        }

        // Increase constraint index
        idx += b.Get().Dimension();

        // Add to map
        jacobian_data_map_[b.id()] = indices;
    }

    // Compress Jacobian
    J.makeCompressed();
    constraint_jacobian_cache_.makeCompressed();

    // With created Jacobian, extract each binding's Jacobian data entries in
    // the data vector in CCS
    for (int i = 0; i < J.nonZeros(); ++i) {
        binding_index_data_t data = *J.valuePtr()[i];
        // Add to the map
        jacobian_data_map_[data[0]][data[1]][data[2]] = i;
    }
}

void SparseSolver::ConstructSparseLagrangianHessian(bool with_constraints) {
    SparseProgram& program = GetCurrentProgram();
    // Create default data structure to hold binding id, jacobian idx and data
    // idx
    typedef Eigen::Vector3i binding_index_data_t;
    int nx = program.NumberOfDecisionVariables();
    // Create sparse Jacobian
    Eigen::SparseMatrix<std::shared_ptr<binding_index_data_t>> H;
    lagrangian_hes_cache_.resize(nx, nx);
    H.resize(nx, nx);
    // Make sure there's no existing data in the sparse matrix
    H.setZero();
    H.data().squeeze();

    lagrangian_hes_cache_.setZero();
    lagrangian_hes_cache_.data().squeeze();

    int idx = 0;
    for (Binding<CostType>& b : program.GetAllCostBindings()) {
        // Create data within the map
        std::vector<std::vector<int>> indices(b.NumberOfVariables());
        // Add constraint to jacobian
        for (int i = 0; i < b.GetVariables().size(); ++i) {
            for (int j = 0; j < b.GetVariables().size(); ++j) {
                // Get sparse Jacobian
                const Eigen::SparseMatrix<double>& Ji = b.Get().Hessian(i, j);
                int cnt = 0;
                // Loop through non-zero entries
                for (int k = 0; k < Ji.outerSize(); ++k) {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(Ji, k);
                         it; ++it) {
                        int id = b.id();
                        // Get location of the non-zero entry
                        int jac_idx = i;
                        // Constraint index
                        int c_idx = idx + it.row();
                        // Index for variable
                        int x_idx = program.GetDecisionVariableIndex(
                            b.GetVariable(i)[it.col()]);
                        // Set element in the full Jacobian to the information
                        // for the Jacobian block provided
                        H.coeffRef(c_idx, x_idx) =
                            std::make_shared<binding_index_data_t>(id, jac_idx,
                                                                   cnt);
                        // Set structure for Jacobian cache
                        constraint_jacobian_cache_.coeffRef(c_idx, x_idx) = 0.0;
                        // Increase data array counter
                        cnt++;
                    }
                }

                indices[i].resize(Ji.nonZeros());
            }
        }

        // Add to map
        jacobian_data_map_[b.id()] = indices;
    }

    // Compress Jacobian
    H.makeCompressed();
    constraint_jacobian_cache_.makeCompressed();

    // With created Jacobian, extract each binding's Jacobian data entries in
    // the data vector in CCS
    for (int i = 0; i < H.nonZeros(); ++i) {
        binding_index_data_t data = *J.valuePtr()[i];
        // Add to the map
        jacobian_data_map_[data[0]][data[1]][data[2]] = i;
    }
}

void SparseSolver::EvaluateCost(Binding<CostType>& binding,
                                const Eigen::VectorXd& x, bool grd, bool hes,
                                bool update_cache) {
    // Get binding inputs
    const Binding<CostType>::VariablePtrVector& var =
        binding.GetVariables();
    const Binding<CostType>::ParameterPtrVector& par =
        binding.GetParameters();
    int nv = var.size();
    int np = par.size();
    // Optional creation of vectors for inputs to the functions (if vector input
    // is not continuous in optimisation vector)
    common::InputRefVector x_in = {}, p_in = {};
    // Extract the input for the binding
    const std::vector<Eigen::Map<Eigen::VectorXd>>& mdata =
        UpdateBindingInputData(binding, x);

    // Set variables
    for (int i = 0; i < np; ++i) {
        x_in.push_back(mdata[i]);
    }
    // Set parameters
    for (int i = 0; i < np; ++i) {
        p_in.push_back(GetCurrentProgram().GetParameterValues(*par[i]));
    }

    const CostType& cost = binding.Get();
    // Evaluate the constraint
    cost.eval(x_in, p_in, grd);
    // if(hes) constraint.eval_hessian(x_in, p_in, l_in)

    if (update_cache == false) return;

    objective_cache_ += cost.ObjectiveFunction()->getOutput(0);

    if (grd) {
        for (int i = 0; i < nv; ++i) {
            UpdateVectorAtVariableLocations(
                objective_gradient_cache_,
                cost.GradientFunction()->getOutput(i), *var[i], continuous[i]);
        }
    }

    if (hes) {
        int cnt = 0;
        // For each Hessian, add to the sparse Hessian
    }
}

void SparseSolver::EvaluateConstraint(Binding<ConstraintType>& binding,
                                      const int& constraint_idx,
                                      const Eigen::VectorXd& x, bool jac,
                                      bool update_cache) {
    // Get binding inputs
    const Binding<ConstraintType>::VariablePtrVector& var =
        binding.GetVariables();
    const Binding<ConstraintType>::ParameterPtrVector& par =
        binding.GetParameters();
    int nv = var.size();
    int np = par.size();
    // Optional creation of vectors for inputs to the functions (if vector input
    // is not continuous in optimisation vector)
    common::InputRefVector x_in = {}, p_in = {};
    
    // Extract the input for the binding
    const std::vector<Eigen::Map<Eigen::VectorXd>>& mdata =
        UpdateBindingInputData(binding, x);

    // Set variables
    for (int i = 0; i < np; ++i) {
        x_in.push_back(mdata[i]);
    }
    // Set parameters
    for (int i = 0; i < np; ++i) {
        p_in.push_back(GetCurrentProgram().GetParameterValues(*par[i]));
    }
    const ConstraintType& constraint = binding.Get();
    // Evaluate the constraint
    constraint.eval(x_in, p_in, jac);
    // if(hes) constraint.eval_hessian(x_in, p_in, l_in)

    if (update_cache == false) return;

    constraint_cache_.middleRows(constraint_idx, constraint.Dimension()) =
        constraint.Vector();

    if (jac) {
        for (int i = 0; i < nv; ++i) {
            VLOG(10) << "Adding Jacobian";
            VLOG(10) << binding.Get().Jacobian(i);
            // Update sparse jacobian data
            for (int j = 0; j < binding.Get().Jacobian(i).nonZeros(); ++j) {
                int idx = jacobian_data_map_[binding.id()][i][j];
                VLOG(10) << "idx = " << idx;
                this->constraint_jacobian_cache_.valuePtr()[idx] =
                    binding.Get().Jacobian(i).valuePtr()[j];
            }
            VLOG(10) << "J =  ";
            VLOG(10) << this->constraint_jacobian_cache_;
        }
    }
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion