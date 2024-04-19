#include "utils/eigen_wrapper.h"

namespace damotion {
namespace utils {
namespace casadi {

Eigen::SparseMatrix<double> CreateSparseEigenMatrix(
    const ::casadi::Sparsity& sparsity, std::vector<casadi_int>& rows,
    std::vector<casadi_int>& cols) {
    // Create Eigen::SparseMatrix from sparsity information
    Eigen::SparseMatrix<double> M(sparsity.rows(), sparsity.columns());

    // Create triplets from data
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.resize(sparsity.nnz());

    // Set triplets initialised with zero value
    for (int k = 0; k < sparsity.nnz(); ++k) {
        triplets[k] = Eigen::Triplet<double>(rows[k], cols[k]);
    }

    // Create matrix from triplets
    M.setFromTriplets(triplets.begin(), triplets.end());

    return M;
}

}  // namespace casadi
}  // namespace utils
}  // namespace damotion