#include "solvers/solver.h"

namespace damotion {
namespace optimisation {
namespace solvers {

void Solver::EvaluateCost(Binding<CostType>& binding, const Eigen::VectorXd& x,
                          bool grd, bool hes, bool update_cache) {
    // Get binding inputs
    const std::vector<sym::VariableVector>& var = binding.GetVariables();
    const sym::ParameterVector& par = binding.GetParameters();
    int nv = var.size();
    int np = par.size();
    // Optional creation of vectors for inputs to the functions (if vector input
    // is not continuous in optimisation vector)
    common::InputRefVector inputs = {};
    // Check if the binding input vectors are continuous within the optimisation
    // vector
    const std::vector<bool> continuous =
        CostBindingContinuousInputCheck(binding);

    // Mapped vectors from existing data
    std::vector<Eigen::Map<const Eigen::VectorXd>> m_vecs = {};
    // Vectors created manually
    std::vector<Eigen::VectorXd> vecs = {};

    // Set variables
    for (int i = 0; i < nv; ++i) {
        if (continuous[i]) {
            // Set input to the start of the vector input
            m_vecs.push_back(Eigen::Map<const Eigen::VectorXd>(
                x.data() +
                    GetCurrentProgram().GetDecisionVariableIndex(var[i][0]),
                var[i].size()));
            inputs.push_back(m_vecs.back());
        } else {
            // Construct a vector for this input
            Eigen::VectorXd xi(var[i].size());
            for (int ii = 0; ii < var[i].size(); ++ii) {
                xi[ii] =
                    x[GetCurrentProgram().GetDecisionVariableIndex(var[i][ii])];
            }
            vecs.push_back(xi);
            inputs.push_back(vecs.back());
        }
    }

    // Set parameters
    for (int i = 0; i < np; ++i) {
        inputs.push_back(GetCurrentProgram().GetParameterValues(par[i]));
    }

    CostType& cost = binding.Get();

    // Check if gradient exists
    if (grd && !cost.HasGradient()) {
        throw std::runtime_error("Cost does not have a gradient!");
    }
    // Check if hessian exists
    if (hes && !cost.HasHessian()) {
        throw std::runtime_error("Cost does not have a hessian!");
    }

    cost.ObjectiveFunction()->call(inputs);
    if (grd) cost.GradientFunction()->call(inputs);
    if (hes) cost.HessianFunction()->call(inputs);

    if (update_cache == false) return;

    objective_cache_ += cost.ObjectiveFunction()->getOutput(0);

    if (grd) {
        for (int i = 0; i < nv; ++i) {
            UpdateVectorAtVariableLocations(
                objective_gradient_cache_,
                cost.GradientFunction()->getOutput(i), var[i], continuous[i]);
        }
    }

    if (hes) {
        int cnt = 0;
        for (int i = 0; i < nv; ++i) {
            for (int j = i; j < nv; ++j) {
                // Increase the count for the Hessian
                UpdateHessianAtVariableLocations(
                    lagrangian_hes_cache_,
                    cost.HessianFunction()->getOutput(cnt), var[i], var[j],
                    continuous[i], continuous[j]);
                cnt++;
            }
        }
    }
}

void Solver::EvaluateCosts(const Eigen::VectorXd& x, bool grad, bool hes) {
    // Reset objective
    objective_cache_ = 0.0;
    // Reset objective gradient
    if (grad) {
        objective_gradient_cache_.setZero();
    }
    if (hes) {
        lagrangian_hes_cache_.setZero();
    }
    // Evaluate all costs in the program
    for (Binding<CostType>& b : GetCosts()) {
        EvaluateCost(b, x, grad, hes);
    }
}

// Evaluates the constraint and updates the cache for the gradients
void Solver::EvaluateConstraint(Binding<ConstraintType>& binding,
                                const int& constraint_idx,
                                const Eigen::VectorXd& x, bool jac,
                                bool update_cache) {
    // Get underlying constraint
    ConstraintType& c = binding.Get();

    // Get size of constraint
    int nc = c.Dimension();

    const std::vector<sym::VariableVector>& var = binding.GetVariables();
    const sym::ParameterVector& par = binding.GetParameters();
    int nv = var.size();
    int np = par.size();

    common::InputRefVector inputs;

    // Check if the binding input vectors are continuous within the optimisation
    // vector
    const std::vector<bool> continuous =
        CostBindingContinuousInputCheck(binding);

    // Mapped vectors from existing data
    std::vector<Eigen::Map<const Eigen::VectorXd>> m_vecs = {};
    // Vectors created manually
    std::vector<Eigen::VectorXd> vecs = {};

    // Set variables
    for (int i = 0; i < nv; ++i) {
        if (continuous[i]) {
            // Set input to the start of the vector input
            m_vecs.push_back(Eigen::Map<const Eigen::VectorXd>(
                x.data() +
                    GetCurrentProgram().GetDecisionVariableIndex(var[i][0]),
                var.size()));
            inputs.push_back(m_vecs.back());
        } else {
            // Construct a vector for this input
            Eigen::VectorXd xi(var[i].size());
            for (int ii = 0; ii < var[i].size(); ++ii) {
                xi[ii] =
                    x[GetCurrentProgram().GetDecisionVariableIndex(var[i][ii])];
            }
            vecs.push_back(xi);
            inputs.push_back(vecs.back());
        }
    }

    // Set parameters
    for (int i = 0; i < np; ++i) {
        inputs.push_back(GetCurrentProgram().GetParameterValues(par[i]));
    }

    // Check if gradient exists
    if (jac && !c.HasJacobian()) {
        throw std::runtime_error("Constraint does not have a Jacobian!");
    }

    c.ConstraintFunction()->call(inputs);
    if (jac) c.JacobianFunction()->call(inputs);

    if (update_cache == false) return;

    constraint_cache_.middleRows(constraint_idx, nc) =
        c.ConstraintFunction()->getOutput(0);

    // Update Jacobian blocks
    for (int i = 0; i < nv; ++i) {
        UpdateJacobianAtVariableLocations(
            constraint_jacobian_cache_, constraint_idx,
            c.JacobianFunction()->getOutput(i), var[i], continuous[i]);
    }
}

void Solver::EvaluateConstraints(const Eigen::VectorXd& x, bool jac) {
    // Reset constraint vector
    constraint_cache_.setZero();
    // Reset objective gradient
    if (jac) {
        constraint_jacobian_cache_.setZero();
    }
    // Loop through all constraints
    for (Binding<ConstraintType>& b : GetConstraints()) {
        EvaluateConstraint(b, 0, x, jac);
    }
}

void Solver::UpdateJacobianAtVariableLocations(Eigen::MatrixXd& jac,
                                               int row_idx,
                                               const Eigen::MatrixXd& block,
                                               const sym::VariableVector& var,
                                               bool is_block) {
    Eigen::Block<Eigen::MatrixXd> J = jac.middleRows(row_idx, block.rows());
    if (is_block) {
        J.middleCols(GetCurrentProgram().GetDecisionVariableIndex(var[0]),
                     var.size()) += block;
    } else {
        // For each variable, update the location in the Jacobian
        for (int j = 0; j < var.size(); ++j) {
            int idx = GetCurrentProgram().GetDecisionVariableIndex(var[j]);
            J.col(idx) += block.col(j);
        }
    }
}

void Solver::UpdateHessianAtVariableLocations(Eigen::MatrixXd& hes,
                                              const Eigen::MatrixXd& block,
                                              const sym::VariableVector& var_x,
                                              const sym::VariableVector& var_y,
                                              bool is_block_x,
                                              bool is_block_y) {
    // For each variable combination
    if (is_block_x && is_block_y) {
        int i_idx = GetCurrentProgram().GetDecisionVariableIndex(var_x[0]),
            j_idx = GetCurrentProgram().GetDecisionVariableIndex(var_y[0]);
        int i_sz = var_x.size(), j_sz = var_y.size();
        // Create lower triangular Hessian
        if (i_idx > j_idx) {
            // Block fill
            hes.block(i_idx, j_idx, i_sz, j_sz) += block;
        } else {
            hes.block(j_idx, i_idx, j_sz, i_sz) += block.transpose();
        }

    } else {
        // For each variable pair, populate the Hessian
        for (int ii = 0; ii < var_x.size(); ++ii) {
            for (int jj = 0; jj < var_y.size(); ++jj) {
                int i_idx =
                        GetCurrentProgram().GetDecisionVariableIndex(var_x[ii]),
                    j_idx =
                        GetCurrentProgram().GetDecisionVariableIndex(var_y[jj]);
                // Create lower triangular matrix
                if (i_idx > j_idx) {
                    hes(i_idx, j_idx) += block(ii, jj);
                } else {
                    hes(j_idx, i_idx) += block(ii, jj);
                }
            }
        }
    }
}

void SparseSolver::ConstructSparseConstraintJacobian() {
    Program& program = GetCurrentProgram();
    typedef Eigen::Vector3i binding_index_data_t;
    Eigen::SparseMatrix<binding_index_data_t> J;
    LOG(INFO) << "Here\n";
    int idx = 0;
    for (auto& b : program.GetAllConstraints()) {
        // Create data within the map
        std::vector<std::vector<int>> indices(b.NumberOfVariables());
        // Add constraint to jacobian
        for (int i = 0; i < b.GetVariables().size(); ++i) {
            Eigen::SparseMatrix<double> Ji =
                b.Get().JacobianFunction()->getOutputSparse(i);
            LOG(INFO) << Ji;
            int cnt = 0;
            // Loop through non-zero entries
            for (int k = 0; k < Ji.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(Ji, k); it;
                     ++it) {
                    // Get location of the non-zero entry
                    int id = b.id();
                    int jac_idx = i;
                    J.coeffRef(idx + it.row(), idx) =
                        binding_index_data_t(id, jac_idx, cnt);
                    // Increase data array counter
                    cnt++;
                }
            }

            indices[i].resize(Ji.nonZeros());
        }

        // Add to map
        sparse_jac_binding_idx_[b.id()] = indices;
    }

    // With created Jacobian, extract each binding's Jacobian data entries in
    // the data vector in CCS
    for (int i = 0; i < J.nonZeros(); ++i) {
        // Create new entry if it doesn't exist
        binding_index_data_t data = J.valuePtr()[i];
        // Add to the map
        sparse_jac_binding_idx_[data[0]][data[1]][data[2]] = i;
    }
}

void SparseSolver::EvaluateCost(Binding<CostType>& binding,
                                const Eigen::VectorXd& x, bool grd, bool hes,
                                bool update_cache) {
    // Get binding inputs
    const std::vector<sym::VariableVector>& var = binding.GetVariables();
    const sym::ParameterVector& par = binding.GetParameters();
    int nv = var.size();
    int np = par.size();
    // Optional creation of vectors for inputs to the functions (if vector input
    // is not continuous in optimisation vector)
    common::InputRefVector inputs = {};
    // Check if the binding input vectors are continuous within the optimisation
    // vector
    const std::vector<bool> continuous =
        CostBindingContinuousInputCheck(binding);

    // Mapped vectors from existing data
    std::vector<Eigen::Map<const Eigen::VectorXd>> m_vecs = {};
    // Vectors created manually
    std::vector<Eigen::VectorXd> vecs = {};

    // Set variables
    for (int i = 0; i < nv; ++i) {
        if (continuous[i]) {
            // Set input to the start of the vector input
            m_vecs.push_back(Eigen::Map<const Eigen::VectorXd>(
                x.data() +
                    GetCurrentProgram().GetDecisionVariableIndex(var[i][0]),
                var[i].size()));
            inputs.push_back(m_vecs.back());
        } else {
            // Construct a vector for this input
            Eigen::VectorXd xi(var[i].size());
            for (int ii = 0; ii < var[i].size(); ++ii) {
                xi[ii] =
                    x[GetCurrentProgram().GetDecisionVariableIndex(var[i][ii])];
            }
            vecs.push_back(xi);
            inputs.push_back(vecs.back());
        }
    }

    // Set parameters
    for (int i = 0; i < np; ++i) {
        inputs.push_back(GetCurrentProgram().GetParameterValues(par[i]));
    }

    CostType& cost = binding.Get();

    // Check if gradient exists
    if (grd && !cost.HasGradient()) {
        throw std::runtime_error("Cost does not have a gradient!");
    }
    // Check if hessian exists
    if (hes && !cost.HasHessian()) {
        throw std::runtime_error("Cost does not have a hessian!");
    }

    cost.ObjectiveFunction()->call(inputs);
    if (grd) cost.GradientFunction()->call(inputs);
    if (hes) cost.HessianFunction()->call(inputs);

    if (update_cache == false) return;

    objective_cache_ += cost.ObjectiveFunction()->getOutput(0);

    if (grd) {
        for (int i = 0; i < nv; ++i) {
            UpdateVectorAtVariableLocations(
                objective_gradient_cache_,
                cost.GradientFunction()->getOutput(i), var[i], continuous[i]);
        }
    }

    if (hes) {
        int cnt = 0;
        for (int i = 0; i < nv; ++i) {
            for (int j = i; j < nv; ++j) {
                // Increase the count for the Hessian
                UpdateHessianAtVariableLocations(
                    lagrangian_hes_cache_,
                    cost.HessianFunction()->getOutput(cnt), var[i], var[j],
                    continuous[i], continuous[j]);
                cnt++;
            }
        }
    }
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion