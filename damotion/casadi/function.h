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
 * @brief Creates a function that given variables x, parameters p, computes the
 * expressions f and optionally their derivatives and hessians. This function is
 * of the form f(x, p, l, der, hes).
 *
 * @param f
 * @param x
 * @param p
 * @return ::casadi::Function
 */
::casadi::Function CreateFunction(const ::casadi::SXVector &f,
                                  const ::casadi::SXVector &x,
                                  const ::casadi::SXVector &p,
                                  const bool &sparse = false) {
  // Create concatenated x vector for derivative purposes
  ::casadi::SX xv = ::casadi::SX::vertcat(x);
  // Flags for whether to compute derivative data
  ::casadi::SX d = ::casadi::SX::sym("derivative"),
               h = ::casadi::SX::sym("hessian");

  // Create input and output vectors
  ::casadi::SXVector in = {}, out = {};
  for (const auto &xi : x) in.push_back(xi);
  for (const auto &pi : p) in.push_back(pi);

  // Organise vector such that it is of the form [f, df, ddf, g, dg, ddg, ...]
  for (const auto &fi : f) {
    out.push_back(fi);

    ::casadi::SX dfi;
    if (fi.size1() == 1) {
      dfi = ::casadi::SX::gradient(f, xv);
    } else {
      dfi = ::casadi::SX::jacobian(f, xv);
    }
    // Densify if requested
    if (!sparse) dfi = ::casadi::SX::densify(dfi);
    out.push_back(::casadi::SX::if_else_zero(d, dfi));

    ::casadi::SX hfi;
    ::casadi::SX li = ::casadi::SX::sym("l", fi.size1());
    // Add these multipliers to the input parameters
    in.push_back(li);
    // Compute lower-triangular hessian matrix
    hfi = ::casadi::SX::tril(::casadi::SX::hessian(mtimes(li.T(), fi), xv));
    // Densify if requested
    if (!sparse) hfi = ::casadi::SX::densify(hfi);
    out.push_back(::casadi::SX::if_else_zero(h, hfi));
  }

  // Append derivative flags at the end of the inputs
  in.push_back(d);
  in.push_back(h);

  // Create function
  return ::casadi::Function("f", in, out);
}

/**
 * @brief Function wrapper base class for casadi functions to Eigen
 * representation
 *
 */
class FunctionWrapper : public common::Function {
  // Forward declarations
 protected:
  struct FunctionWrapperData;

 public:
  using SharedPtr = std::shared_ptr<FunctionWrapper>;
  using UniquePtr = std::unique_ptr<FunctionWrapper>;

  FunctionWrapper() = default;

  /**
   * @brief Constructs a new FunctionWrapper without an expression, but
   * initialises the function for the provided dimensions.
   *
   * @param nx
   * @param ny
   * @param np
   */
  FunctionWrapper(const int &nx, const int &ny, const int &np = 0)
      : common::Function(nx, ny) {}

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
  void CreateFunctionData(const ::casadi::Function &f,
                          FunctionWrapperData &data,
                          std::vector<GenericEigenMatrix> &out) {
    if (f.is_null()) {
      return;
    }

    // TODO - See if this has a negative effect
    f_ = std::move(f);

    // Initialise output data
    out = {};
    out_data_ptr_ = {};

    // Checkout memory object for function
    mem_ = f.checkout();

    iw_.assign(f.sz_iw(), 0);
    dw_.assign(f.sz_w(), 0.0);

    // Create dense matrices for the output
    for (int i = 0; i < f.n_out(); ++i) {
      const ::casadi::Sparsity &sparsity = f.sparsity_out(i);
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
      out.push_back(GenericEigenMatrix(M));
      out_data_ptr_.push_back(out.back().data());

      VLOG(10) << f.name() << " Dense Output " << i;
      VLOG(10) << out.back();
    }
  }

  /**
   * @brief Calls the function with the current inputs
   *
   */
  void EvalImpl(const common::InputRefVector &input,
                bool check = false) override {
    // TODO - If performing a check
    // Set inputs
    std::vector<const double *> in_data_ptr = {};
    for (const auto &in : input) {
      in_data_ptr.push_back(in.data());
    }

    // Call the function using the currently set inputs
    f_(in_data_ptr.data(), out_data_ptr_.data(), iw_.data(), dw_.data(), mem_);
  }

  /**
   * @copydoc GenericEigenMatrix::getOutput()
   *
   * @param i
   * @return const GenericEigenMatrix&
   */
  const GenericEigenMatrix &GetOutputImpl(const size_t &i) const override {
    return f_out_[i];
  }

 protected:
  // Data output pointers for casadi function
  std::vector<double *> out_data_ptr_;

  // Row triplet data for nnz of each output
  std::vector<std::vector<casadi_int>> rows_;
  // Column triplet data for nnz of each output
  std::vector<std::vector<casadi_int>> cols_;

  // Memory allocated for function evaluation
  int mem_;

  // Integer working vector
  std::vector<casadi_int> iw_;
  // Double working vector
  std::vector<double> dw_;

  // Function
  mutable ::casadi::Function f_;

 private:
  // Output matrix data
  std::vector<GenericEigenMatrix> f_out_;
};

}  // namespace casadi
}  // namespace damotion

#endif /* CASADI_FUNCTION_H */
