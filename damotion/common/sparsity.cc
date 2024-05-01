#include "damotion/common/sparsity.h"

namespace damotion {
namespace common {

Eigen::SparseMatrix<double> CreateSparseEigenMatrix(Sparsity &sparsity) {
    // Create Eigen::SparseMatrix from sparsity information
    Eigen::SparseMatrix<double> M(sparsity.rows(), sparsity.cols());

    // Create triplets from data
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.resize(sparsity.nnz());

    std::vector<long long> rows, cols;
    sparsity.GetTriplets(rows, cols);

    // Set triplets initialised with zero value
    for (int k = 0; k < sparsity.nnz(); ++k) {
        triplets[k] = Eigen::Triplet<double>(rows[k], cols[k]);
    }

    // Create matrix from triplets
    M.setFromTriplets(triplets.begin(), triplets.end());

    return M;
}

}  // namespace common
}  // namespace damotion
