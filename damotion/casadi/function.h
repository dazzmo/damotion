#ifndef CASADI_FUNCTION_H
#define CASADI_FUNCTION_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <casadi/casadi.hpp>
#include <pinocchio/autodiff/casadi.hpp>

#include "damotion/casadi/codegen.h"
#include "damotion/common/function.h"
#include "damotion/common/sparsity.h"

namespace damotion {
namespace casadi {

/**
 * @brief Function wrapper base class for casadi functions to Eigen
 * representation
 *
 */
template <typename MatrixType>
class FunctionWrapper : public common::Function<MatrixType> {
 public:
  using SharedPtr = std::shared_ptr<FunctionWrapper<MatrixType>>;

  FunctionWrapper() = default;
  ~FunctionWrapper() {
    // Release memory for casadi function
    if (!f_.is_null()) {
      f_.release(mem_);
    }
  }

  FunctionWrapper(const ::casadi::Function &f)
      : common::Function<MatrixType>(f.n_in(), f.n_out()) {
    *this = f;
  }

  FunctionWrapper(const FunctionWrapper &other) { *this = other.f_; }

  FunctionWrapper &operator=(::casadi::Function f) {
    if (f.is_null()) {
      return *this;
    }

    // Copy function
    this->f_ = f;

    this->SetNumberOfInputs(f.n_in());
    this->SetNumberOfOutputs(f.n_out());

    // Initialise output data
    this->OutputVector() = {};
    this->out_data_ptr_ = {};

    // Checkout memory object for function
    this->mem_ = f.checkout();

    // Resize work vectors
    this->in_data_ptr_.assign(f_.sz_arg(), nullptr);

    this->iw_.assign(f_.sz_iw(), 0);
    this->dw_.assign(f_.sz_w(), 0.0);

    // Create dense matrices for the output
    for (int i = 0; i < f_.n_out(); ++i) {
      const ::casadi::Sparsity &sparsity = f_.sparsity_out(i);
      // Create dense matrix for output and add data to output data
      // pointer vector
      this->OutputVector().push_back(
          Eigen::MatrixXd::Zero(sparsity.rows(), sparsity.columns()));
      this->out_data_ptr_.push_back(this->OutputVector().back().data());

      VLOG(10) << f.name() << " Dense Output " << i;
      VLOG(10) << this->OutputVector().back();
    }

    return *this;
  }

  FunctionWrapper &operator=(const FunctionWrapper &other) {
    *this = other.f_;
    return *this;
  }

  /**
   * @brief The casadi::Function that is wrapped.
   *
   * @return casadi::Function&
   */
  ::casadi::Function &f() { return f_; }

  /**
   * @brief Calls the function with the current inputs
   *
   */
  void callImpl(const common::InputRefVector &input) override {
    // Set vector of inputs
    int idx = 0;
    for (const Eigen::Ref<const Eigen::VectorXd> &in : input) {
      in_data_ptr_[idx++] = in.data();
    }
    // Call the function
    f_(in_data_ptr_.data(), out_data_ptr_.data(), iw_.data(), dw_.data(), mem_);
  }

 protected:
  // Data input vector for casadi function
  mutable std::vector<const double *> in_data_ptr_;
  // Data output pointers for casadi function
  mutable std::vector<double *> out_data_ptr_;

  // Row triplet data for nnz of each output
  std::vector<std::vector<casadi_int>> rows_;
  // Column triplet data for nnz of each output
  std::vector<std::vector<casadi_int>> cols_;

  // Memory allocated for function evaluation
  int mem_;

  // Integer working vector
  mutable std::vector<casadi_int> iw_;
  // Double working vector
  mutable std::vector<double> dw_;

  // Underlying function
  mutable ::casadi::Function f_;
};

/**
 * @brief Factory for generating functions based on a casadi::Function object.
 *
 * @tparam MatrixType
 * @param f
 * @param codegen
 * @param dir
 * @return common::Function<MatrixType>::SharedPtr
 */
template <typename MatrixType>
typename common::Function<MatrixType>::SharedPtr FunctionFactory(
    const ::casadi::Function &f, bool codegen = false,
    const std::string &dir = "./") {
  // Create new FunctionWrapper
  typename FunctionWrapper<MatrixType>::SharedPtr fptr =
      std::make_shared<FunctionWrapper<MatrixType>>(f);
  // Generate code for function, if requested
  if (codegen) {
    fptr->f() = damotion::casadi::codegen(fptr->f(), dir);
  }
  // Return as Function<MatrixType>::SharedPtr
  return fptr;
}

// Class specialisations
template <>
FunctionWrapper<double> &FunctionWrapper<double>::operator=(
    ::casadi::Function f);

template <>
FunctionWrapper<Eigen::SparseMatrix<double>> &
FunctionWrapper<Eigen::SparseMatrix<double>>::operator=(::casadi::Function f);

}  // namespace casadi
}  // namespace damotion

#endif /* CASADI_FUNCTION_H */
