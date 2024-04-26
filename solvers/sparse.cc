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
            const Eigen::SparseMatrix<double>& Ji =
                b.Get().JacobianFunction()->getOutput(i);
            int cnt = 0;
            // Loop through non-zero entries
            for (int k = 0; k < Ji.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(Ji, k); it;
                     ++it) {
                    // Get location of the non-zero entry
                    int id = b.id();
                    int jac_idx = i;
                    int c_idx = idx + it.row();
                    int x_idx = program.GetDecisionVariableIndex(
                        b.GetVariable(i)[it.col()]);
                    J.coeffRef(c_idx, x_idx) =
                        std::make_shared<binding_index_data_t>(id, jac_idx,
                                                               cnt);
                    constraint_jacobian_cache_.coeffRef(c_idx, x_idx) = 0.0;
                    // Increase data array counter
                    cnt++;
                }
            }

            indices[i].resize(Ji.nonZeros());
        }

        // Add to map
        jacobian_data_map_[b.id()] = indices;
    }

    // Compress Jacobian
    J.makeCompressed();
    constraint_jacobian_cache_.makeCompressed();

    // With created Jacobian, extract each binding's Jacobian data entries in
    // the data vector in CCS
    for (int i = 0; i < J.nonZeros(); ++i) {
        // Create new entry if it doesn't exist
        binding_index_data_t data = *J.valuePtr()[i];
        // Add to the map
        jacobian_data_map_[data[0]][data[1]][data[2]] = i;
    }
}

void SparseSolver::EvaluateCost(Binding<CostType>& binding,
                                const Eigen::VectorXd& x, bool grd, bool hes,
                                bool update_cache) {
    // Get binding inputs
    const Binding<CostType>::VariablePtrVector& var = binding.GetVariables();
    const Binding<CostType>::ParameterPtrVector& par = binding.GetParameters();
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
                    GetCurrentProgram().GetDecisionVariableIndex((*var[i])[0]),
                var[i]->size()));
            inputs.push_back(m_vecs.back());
        } else {
            // Construct a vector for this input
            Eigen::VectorXd xi(var[i]->size());
            for (int ii = 0; ii < var[i]->size(); ++ii) {
                xi[ii] = x[GetCurrentProgram().GetDecisionVariableIndex(
                    (*var[i])[ii])];
            }
            vecs.push_back(xi);
            inputs.push_back(vecs.back());
        }
    }

    // Set parameters
    for (int i = 0; i < np; ++i) {
        inputs.push_back(GetCurrentProgram().GetParameterValues(*par[i]));
    }

    const CostType& cost = binding.Get();

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
                cost.GradientFunction()->getOutput(i), *var[i], continuous[i]);
        }
    }

    if (hes) {
        int cnt = 0;
        // For each Hessian, add to the sparse Hessian
    }
}

void SparseSolver::EvaluateConstraint(const Binding<ConstraintType>& binding,
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
                    GetCurrentProgram().GetDecisionVariableIndex((*var[i])[0]),
                var[i]->size()));
            inputs.push_back(m_vecs.back());
        } else {
            // Construct a vector for this input
            Eigen::VectorXd xi(var[i]->size());
            for (int ii = 0; ii < var[i]->size(); ++ii) {
                xi[ii] = x[GetCurrentProgram().GetDecisionVariableIndex(
                    (*var[i])[ii])];
            }
            vecs.push_back(xi);
            inputs.push_back(vecs.back());
        }
    }

    // Set parameters
    for (int i = 0; i < np; ++i) {
        inputs.push_back(GetCurrentProgram().GetParameterValues(*par[i]));
    }

    const ConstraintType& constraint = binding.Get();

    // Check if gradient exists
    if (jac && !constraint.HasJacobian()) {
        throw std::runtime_error("Constraint does not have a jacobian!");
    }
    // Check if hessian exists
    // if (hes && !cost.HasHessian()) {
    // throw std::runtime_error("Cost does not have a hessian!");
    // }

    constraint.ConstraintFunction()->call(inputs);
    if (jac) constraint.JacobianFunction()->call(inputs);
    // if (hes) constraint.HessianFunction()->call(inputs);

    if (update_cache == false) return;

    constraint_cache_.middleRows(constraint_idx, constraint.Dimension()) =
        constraint.ConstraintFunction()->getOutput(0);

    if (jac) {
        for (int i = 0; i < nv; ++i) {
            LOG(INFO) << "Called Jacobian Ji "
                      << binding.Get().JacobianFunction()->getOutput(i);
            // Update sparse jacobian data
            for (int j = 0; j < var[i]->size(); ++j) {
                int idx = jacobian_data_map_[binding.id()][i][j];
                this->constraint_jacobian_cache_.valuePtr()[i] =
                    binding.Get()
                        .JacobianFunction()
                        ->getOutput(i)
                        .valuePtr()[j];
            }
        }
    }
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion