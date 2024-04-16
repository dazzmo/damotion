#include "utils/eigen_wrapper.h"

namespace damotion {
namespace utils {
namespace casadi {

FunctionWrapper::FunctionWrapper(::casadi::Function f)
    : Function(f.n_in(), f.n_out()) {
    *this = f;
}

FunctionWrapper::FunctionWrapper(const FunctionWrapper& other) {
    *this = other.f_;
}

FunctionWrapper& FunctionWrapper::operator=(const FunctionWrapper& other) {
    *this = other.f_;
    return *this;
}

FunctionWrapper::~FunctionWrapper() {
    // Release memory for casadi function
    if (!f_.is_null()) {
        f_.release(mem_);
    }
}

FunctionWrapper& FunctionWrapper::operator=(::casadi::Function f) {
    if (f.is_null()) {
        return *this;
    }

    // Copy function
    f_ = f;

    // Checkout memory object for function
    mem_ = f.checkout();

    // Resize work vectors
    in_data_ptr_.assign(f_.sz_arg(), nullptr);

    iw_.assign(f_.sz_iw(), 0);
    dw_.assign(f_.sz_w(), 0.0);

    // Set all outputs to dense
    is_out_sparse_ = std::vector<bool>(f_.n_out(), false);

    // Create dense matrices for the output
    for (int i = 0; i < f_.n_out(); ++i) {
        const ::casadi::Sparsity& sparsity = f_.sparsity_out(i);
        // Create dense matrix for output and add data to output data pointer
        // vector
        out_.push_back(Eigen::MatrixXd(sparsity.rows(), sparsity.columns()));
        out_data_ptr_.push_back(out_.back().data());

        // Get sparsity data for matrix
        std::vector<casadi_int> rows, cols;
        sparsity.get_triplet(rows, cols);
        rows_.push_back(rows);
        cols_.push_back(cols);
        // Create emptry sparse matrix
        out_sparse_.push_back(Eigen::SparseMatrix<double>(0, 0));
    }

    return *this;
}

void FunctionWrapper::setSparseOutput(int i) {
    // Set output flag to true and remove dense matrix data
    is_out_sparse_[i] = true;
    out_[i].resize(0, 0);
    // Create sparsity pattern for the output
    out_sparse_[i] = createSparseMatrix(f_.sparsity_out(i), rows_[i], cols_[i]);
    // Replace data pointer with sparse output
    out_data_ptr_[i] = out_sparse_[i].valuePtr();
}

void FunctionWrapper::callImpl(const Function::InputRefVector& input) {
    // Set vector of inputs
    int idx = 0;
    for (const Eigen::Ref<const Eigen::VectorXd>& x : input) {
        in_data_ptr_[idx++] = x.data();
    }
    // Call the function
    f_(in_data_ptr_.data(), out_data_ptr_.data(), iw_.data(), dw_.data(), mem_);
}

Eigen::SparseMatrix<double> FunctionWrapper::createSparseMatrix(
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