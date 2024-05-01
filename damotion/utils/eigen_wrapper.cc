#include "damotion/utils/eigen_wrapper.h"

namespace damotion {
namespace utils {
namespace casadi {

FunctionWrapper::FunctionWrapper(::casadi::Function f) { *this = f; }

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
    // Get sparsity data for matrix
    std::vector<casadi_int> rows, cols;
    sparsity.get_triplet(rows, cols);
    rows_.push_back(rows);
    cols_.push_back(cols);
    // Assign data for output
    out_data_.emplace_back(sparsity.nnz(), 0);
    out_data_ptr_.push_back(out_data_.back().data());
    // Create dense matrix
    out_.push_back(Eigen::MatrixXd(sparsity.rows(), sparsity.columns()));
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
}

void FunctionWrapper::setInput(int i, Eigen::Ref<const Eigen::MatrixXd> x) {
  // Check input dimension
  if (x.size() != f_.size1_in(i)) {
    throw std::invalid_argument(f_.name() + ": input " + std::to_string(i) +
                                " is incorrect dimension");
  }

  // Check if all values are good
  // TODO: Add flag to perform these checks if runtime speed is critical
  if (x.hasNaN() || !x.allFinite()) {
    std::ostringstream ss;
    ss << f_.name() + ": input " << i << " has invalid values:\n"
       << x.transpose().format(3);
    throw std::runtime_error(ss.str());
  }

  // Otherwise, add vector data pointer to input
  in_data_ptr_[i] = x.data();
}

void FunctionWrapper::setInput(int i, const double* x_ptr) {
  // TODO - Perform checks of the input data
  in_data_ptr_[i] = x_ptr;
}

void FunctionWrapper::setInput(
    const std::vector<int>& idx,
    const std::vector<Eigen::Ref<const Eigen::MatrixXd>>& x) {
  for (int i = 0; i < idx.size(); ++i) {
    setInput(idx[i], x[i]);
  }
}

void FunctionWrapper::call() {
  // Call the function
  f_(in_data_ptr_.data(), out_data_ptr_.data(), iw_.data(), dw_.data(), mem_);
}

const Eigen::MatrixXd& FunctionWrapper::getOutput(int i) {
  // Use sparse output data to construct dense matrix
  out_[i].setZero();
  const ::casadi::Sparsity& sp = f_.sparsity_out(i);
  // Set non-zero entries in the dense matrix
  for (int k = 0; k < sp.nnz(); ++k) {
    out_[i](rows_[i][k], cols_[i][k]) = out_data_[i][k];
  }
  // Return the dense output i
  return out_[i];
}

const Eigen::SparseMatrix<double>& FunctionWrapper::getOutputSparse(int i) {
  if (is_out_sparse_[i] != true) {
    // ! Throw warning and create sparse output
  }
  // Copy non-zero outputs
  std::copy(out_data_[i].begin(), out_data_[i].end(),
            out_sparse_[i].valuePtr());
  // Return the sparse output i
  return out_sparse_[i];
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
