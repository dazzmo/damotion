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
  using UniquePtr = std::unique_ptr<FunctionWrapper>;

  FunctionWrapper() = default;

  /**
   * @brief Constructs a new FunctionWrapper without an expression, but
   * initialises the function for the provided dimensions.
   *
   * @param n_in
   * @param n_out
   */
  FunctionWrapper(const size_t &n_in, const size_t &n_out)
      : common::Function(n_in, n_out) {}

  /**
   * @brief Creates a function object based on the Casadi symbolic type.
   *
   * @param f Expressions to use as outputs for the function
   * @param x Input variables
   * @param p Input parameters
   * @param derivative Whether to compute the derivative of the expression with
   * respect to x
   * @param hessian Whether to compute the hessian of the expression with
   * respect to x
   */
  FunctionWrapper(const ::casadi::Function &f) : common::Function() {}

  ~FunctionWrapper() {
    // Release memory for casadi functions
    if (!f_.is_null()) f_.release(mem_);
  }

  /**
   * @brief Given a casadi::Function object, creates the data and output matrix
   * data vectors required for evaluation.
   *
   * @param f
   * @param data
   * @param out
   */
  void createFunctionData(const ::casadi::Function &f) {
    if (f.is_null()) {
      return;
    }

    // TODO - See if this has a negative effect
    f_ = std::move(f);

    // Create outputs for the system
    out_.resize(f.n_out());

    // Create Function::Output objects for each input, paying attention to
    // matrix densities
    for (casadi_int i = 0; i < f.n_out(); ++i) {
      const ::casadi::Sparsity &sparsity = f_.sparsity_out(i);

      size_t rows = sparsity.rows();
      size_t cols = sparsity.columns();
      size_t nnz = sparsity.nnz();

      // Create output based on density of matrix
      if (nnz < rows * cols) {
        // Sparse output, get i_row and j_col pointers
        std::vector<casadi_int> r, c;
        sparsity.get_triplet(r, c);
        std::vector<int> i_row = std::vector<int>(r.begin(), r.end());
        std::vector<int> j_col = std::vector<int>(c.begin(), c.end());
        // Create sparse output
        out_[i] = Function::Output(rows, cols, nnz, i_row, j_col);
      } else {
        // Create dense output
        out_[i] = Function::Output(rows, cols);
      }

      // Register data with output vector
      out_data_ptr_.push_back(out_[i].data());
    }

    // Checkout memory object for function
    mem_ = f.checkout();

    iw_.assign(f.sz_iw(), 0);
    dw_.assign(f.sz_w(), 0.0);
  }

  /**
   * @brief Calls the function with the current inputs
   *
   */
  void evalImpl(const common::Function::InputVector &input) override {
    std::vector<const Scalar *> in_data_ptr = {};
    for (const auto &in : input) {
      in_data_ptr.push_back(in.data());
    }
    // Call the function
    f_(in_data_ptr.data(), out_data_ptr_.data(), iw_.data(), dw_.data(), mem_);
  }

 protected:
  // Functions
  mutable ::casadi::Function f_;

  // Vector of data pointers for each output
  std::vector<Scalar *> out_data_ptr_;

 private:
  // Memory allocated for function evaluations
  int mem_;
  // Integer working vectors
  std::vector<casadi_int> iw_;
  // Double working vectors
  std::vector<double> dw_;
};

}  // namespace casadi
}  // namespace damotion

#endif /* CASADI_FUNCTION_H */
