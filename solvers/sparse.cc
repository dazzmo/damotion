#include "solvers/sparse.h"

namespace damotion {
namespace optimisation {
namespace solvers {

void SparseSolver::ConstructSparseConstraintJacobian() {
    SparseProgram& program = GetCurrentProgram();
    typedef Eigen::Vector3i binding_index_data_t;
    Eigen::SparseMatrix<binding_index_data_t> J;
    LOG(INFO) << "Here\n";
    int idx = 0;
    for (Binding<SparseConstraint>& b : program.GetAllConstraintBindings()) {
        // Create data within the map
        std::vector<std::vector<int>> indices(b.NumberOfVariables());
        // Add constraint to jacobian
        for (int i = 0; i < b.GetVariables().size(); ++i) {
            Eigen::SparseMatrix<double> Ji =
                b.Get().JacobianFunction()->getOutput(i);
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
        jacobian_data_map_[b.id()] = indices;
    }

    // With created Jacobian, extract each binding's Jacobian data entries in
    // the data vector in CCS
    for (int i = 0; i < J.nonZeros(); ++i) {
        // Create new entry if it doesn't exist
        binding_index_data_t data = J.valuePtr()[i];
        // Add to the map
        jacobian_data_map_[data[0]][data[1]][data[2]] = i;
    }
}

void SparseSolver::EvaluateCost(Binding<CostType>& binding,
                                const Eigen::VectorXd& x, bool grd, bool hes,
                                bool update_cache) {
    // Get binding inputs
    const std::vector<std::shared_ptr<sym::VariableVector>>& var =
        binding.GetVariables();
    const std::vector<std::shared_ptr<sym::Parameter>>& par =
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

    // if (hes) {
    //     int cnt = 0;
    //     for (int i = 0; i < nv; ++i) {
    //         for (int j = i; j < nv; ++j) {
    //             // Increase the count for the Hessian
    //             UpdateHessianAtVariableLocations(
    //                 lagrangian_hes_cache_,
    //                 cost.HessianFunction()->getOutput(cnt), var[i], var[j],
    //                 continuous[i], continuous[j]);
    //             cnt++;
    //         }
    //     }
    // }
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion