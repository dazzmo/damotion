#include "utils/eigen_wrapper.h"

namespace damotion {
namespace utils {
namespace casadi {

// Class specialisations
template <>
FunctionWrapper<double> &FunctionWrapper<double>::operator=(
    ::casadi::Function f) {
    if (f.is_null()) {
        return *this;
    }

    // Copy function
    f_ = f;

    SetNumberOfInputs(f.n_in());
    SetNumberOfOutputs(f.n_out());

    // Initialise output data
    OutputVector() = {};
    out_data_ptr_ = {};

    // Checkout memory object for function
    mem_ = f.checkout();

    // Resize work vectors
    in_data_ptr_.assign(f_.sz_arg(), nullptr);

    iw_.assign(f_.sz_iw(), 0);
    dw_.assign(f_.sz_w(), 0.0);

    // Create dense matrices for the output
    for (int i = 0; i < f_.n_out(); ++i) {
        const ::casadi::Sparsity &sparsity = f_.sparsity_out(i);
        // Create dense matrix for output and add data to output data pointer
        // vector
        OutputVector().push_back(0.0);
        out_data_ptr_.push_back(&OutputVector().back());
    }

    return *this;
}

template <>
FunctionWrapper<Eigen::SparseMatrix<double>> &
FunctionWrapper<Eigen::SparseMatrix<double>>::operator=(::casadi::Function f) {
    if (f.is_null()) {
        return *this;
    }

    // Copy function
    f_ = f;

    SetNumberOfInputs(f.n_in());
    SetNumberOfOutputs(f.n_out());

    // Initialise output data
    OutputVector() = {};
    out_data_ptr_ = {};

    // Checkout memory object for function
    mem_ = f.checkout();

    // Resize work vectors
    in_data_ptr_.assign(f_.sz_arg(), nullptr);

    iw_.assign(f_.sz_iw(), 0);
    dw_.assign(f_.sz_w(), 0.0);

    // Create dense matrices for the output
    for (int i = 0; i < f_.n_out(); ++i) {
        const ::casadi::Sparsity &sparsity = f_.sparsity_out(i);
        // Get sparsity data for matrix
        std::vector<casadi_int> rows, cols;
        sparsity.get_triplet(rows, cols);
        rows_.push_back(rows);
        cols_.push_back(cols);
        // Create sparse matrix with sparsity structure
        Eigen::SparseMatrix<double> M =
            CreateSparseEigenMatrix(sparsity, rows, cols);
        OutputVector().push_back(M);
        out_data_ptr_.push_back(OutputVector().back().valuePtr());
    }

    return *this;
}

Eigen::SparseMatrix<double> CreateSparseEigenMatrix(
    const ::casadi::Sparsity &sparsity, std::vector<casadi_int> &rows,
    std::vector<casadi_int> &cols) {
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