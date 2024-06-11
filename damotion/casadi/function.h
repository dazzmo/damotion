#ifndef CASADI_FUNCTION_H
#define CASADI_FUNCTION_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <casadi/casadi.hpp>
#include <pinocchio/autodiff/casadi.hpp>

#include "damotion/casadi/codegen.h"
#include "damotion/common/function.h"

namespace damotion {
namespace casadi {

/**
 * @brief Function wrapper base class for casadi functions to Eigen
 * representation
 *
 */
class FunctionWrapper : public common::Function {
 public:
  using SharedPtr = std::shared_ptr<FunctionWrapper>;

  FunctionWrapper() = default;
  ~FunctionWrapper() {
    // Release memory for casadi function
    if (!f_.is_null()) {
      f_.release(mem_);
    }
  }

  FunctionWrapper(const ::casadi::Function &f)
      : common::Function(f.n_in(), f.n_out()) {
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
    out_ = {};
    this->out_data_ptr_ = {};

    // Checkout memory object for function
    this->mem_ = f.checkout();

    this->iw_.assign(f_.sz_iw(), 0);
    this->dw_.assign(f_.sz_w(), 0.0);

    // Create dense matrices for the output
    for (int i = 0; i < f_.n_out(); ++i) {
      const ::casadi::Sparsity &sparsity = f_.sparsity_out(i);
      std::vector<casadi_int> rows = {}, cols = {};
      sparsity.get_triplet(rows, cols);
      // Compute sparse matrix through eigen
      std::vector<Eigen::Triplet<int>> triplets;
      for (int i = 0; i < sparsity.nnz(); ++i) {
        triplets.push_back(Eigen::Triplet<int>(rows[i], cols[i], 0.0));
      }
      Eigen::SparseMatrix<double> M(sparsity.rows(), sparsity.columns());
      M.setFromTriplets(triplets.begin(), triplets.end());
      // Create generic matrix data
      out_.push_back(GenericEigenMatrix(M));
      this->out_data_ptr_.push_back(out_.back().data());

      VLOG(10) << f.name() << " Dense Output " << i;
      VLOG(10) << out_.back();
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
  void callImpl() override {
    // Call the function using the currently set inputs
    f_(in_.data(), out_data_ptr_.data(), iw_.data(), dw_.data(), mem_);
  }

  /**
   * @copydoc GenericEigenMatrix::getOutput()
   *
   * @param i
   * @return const GenericEigenMatrix&
   */
  const GenericEigenMatrix &getOutput(const int &i) const override {
    return out_[i];
  }

 protected:
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

 private:
  // Output matrix data
  std::vector<GenericEigenMatrix> out_;
};

/**
 * @brief Factory for generating functions based on a casadi::Function object.
 *
 * @param f
 * @param codegen
 * @param dir
 * @return common::Function::SharedPtr
 */
typename common::Function::SharedPtr FunctionFactory(
    const ::casadi::Function &f, bool codegen = false,
    const std::string &dir = "./") {
  // Create new FunctionWrapper
  typename FunctionWrapper::SharedPtr fptr =
      std::make_shared<FunctionWrapper>(f);
  // Generate code for function, if requested
  if (codegen) {
    fptr->f() = damotion::casadi::codegen(fptr->f(), dir);
  }
  // Return as Function::SharedPtr
  return fptr;
}

}  // namespace casadi
}  // namespace damotion

#endif /* CASADI_FUNCTION_H */
