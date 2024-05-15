#include "damotion/casadi/function.h"

namespace damotion {
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
    // Create sparse matrix
    common::Sparsity sparsity(f_.sparsity_out(i));
    Eigen::SparseMatrix<double> M = common::CreateSparseEigenMatrix(sparsity);
    OutputVector().push_back(M);
    out_data_ptr_.push_back(OutputVector().back().valuePtr());

    VLOG(10) << f.name() << " Sparse Output " << i;
    VLOG(10) << OutputVector().back();
  }

  return *this;
}

}  // namespace casadi
}  // namespace damotion
