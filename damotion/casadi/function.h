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
class CasadiFunction : public common::Function {
  // Forward declarations
 protected:
  struct CasadiFunctionData;

 public:
  using SharedPtr = std::shared_ptr<CasadiFunction>;
  using UniquePtr = std::unique_ptr<CasadiFunction>;

  CasadiFunction() = default;

  /**
   * @brief Constructs a new CasadiFunction without an expression, but
   * initialises the function for the provided dimensions.
   *
   * @param nx
   * @param ny
   * @param np
   */
  CasadiFunction(const int &nx, const int &ny, const int &np = 0)
      : common::Function(nx, ny, np) {}

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
  CasadiFunction(const ::casadi::SXVector &f, const ::casadi::SXVector &x,
                 const ::casadi::SXVector &p, bool derivative = false,
                 bool hessian = false, bool sparse = false)
      : common::Function(x.size(), p.size(), f.size()) {
    // Create functions to compute the function, derivative and hessian
    // Concatenate x and p
    ::casadi::SXVector in;
    for (const auto &xi : x) in.push_back(xi);
    for (const auto &pi : p) in.push_back(pi);
    ::casadi::Function f_("name", in, f);
    CreateFunctionData(f_, f_data_, f_out_);

    // Create concatenated x vector for derivative purposes
    ::casadi::SX xv = ::casadi::SX::vertcat(x);

    if (derivative) {
      has_derivative_ = true;
      ::casadi::SXVector df = {};
      for (const auto &fi : f) {
        ::casadi::SX dfi;
        if (fi.size1() == 1) {
          dfi = ::casadi::SX::gradient(f, xv);
        } else {
          dfi = ::casadi::SX::jacobian(f, xv);
        }
        // Densify if requested
        if (!sparse) dfi = ::casadi::SX::densify(dfi);
        df.push_back(dfi);
      }
      ::casadi::Function d("name_d", in, {df});
    }

    if (hessian) {
      has_hessian_ = true;
      ::casadi::SXVector hf = {};
      for (const auto &fi : f) {
        ::casadi::SX hfi;
        ::casadi::SX li = ::casadi::SX::sym("l", fi.size1());
        // Compute lower-triangular hessian matrix
        hfi = ::casadi::SX::tril(::casadi::SX::hessian(mtimes(li.T(), fi), xv));
        // Densify if requested
        if (!sparse) hfi = ::casadi::SX::densify(hfi);
        // Create system with lagrangian multipliers
        in.push_back(li);
        hf.push_back(hfi);
      }
      ::casadi::Function d("name_d", in, {hf});
    }
  }

  ~CasadiFunction() {
    // Release memory for casadi functions
    if (!f_.is_null()) f_.release(f_data_.mem_);
    if (!d_.is_null()) d_.release(d_data_.mem_);
    if (!h_.is_null()) h_.release(h_data_.mem_);
  }

  /**
   * @brief Given a casadi::Function object, creates the data and output matrix
   * data vectors required for evaluation.
   *
   * @param f
   * @param data
   * @param out
   */
  void CreateFunctionData(const ::casadi::Function &f, CasadiFunctionData &data,
                          std::vector<GenericEigenMatrix> &out) {
    if (f.is_null()) {
      return;
    }

    // Initialise output data
    out = {};
    data.out_data_ptr_ = {};

    // Checkout memory object for function
    data.mem_ = f.checkout();

    data.iw_.assign(f.sz_iw(), 0);
    data.dw_.assign(f.sz_w(), 0.0);

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
      data.out_data_ptr_.push_back(out.back().data());

      VLOG(10) << f.name() << " Dense Output " << i;
      VLOG(10) << out.back();
    }
  }

  /**
   * @brief Calls the function with the current inputs
   *
   */
  void EvalImpl() override {
    // Call the function using the currently set inputs
    f_(in_.data(), f_data_.out_data_ptr_.data(), f_data_.iw_.data(),
       f_data_.dw_.data(), f_data_.mem_);
  }

  void DerivativeImpl() override {
    // Call the function using the currently set inputs
    d_(in_.data(), d_data_.out_data_ptr_.data(), d_data_.iw_.data(),
       d_data_.dw_.data(), d_data_.mem_);
  }

  void HessianImpl() override {
    // Call the function using the currently set inputs
    h_(in_.data(), h_data_.out_data_ptr_.data(), h_data_.iw_.data(),
       h_data_.dw_.data(), h_data_.mem_);
  }

  /**
   * @copydoc GenericEigenMatrix::getOutput()
   *
   * @param i
   * @return const GenericEigenMatrix&
   */
  const GenericEigenMatrix &GetOutput(const int &i) const override {
    assert(i < ny() && "Number of outputs exceeded");
    return f_out_[i];
  }

  const GenericEigenMatrix &GetDerivative(const int &i) const override {
    assert(i < ny() && "Number of outputs exceeded");
    assert(HasDerivative() && "Derivative is not implemented!");
    return d_out_[i];
  }

  const GenericEigenMatrix &GetHessian(const int &i) const override {
    assert(i < ny() && "Number of outputs exceeded");
    assert(HasHessian() && "Hessian is not implemented!");
    return h_out_[i];
  }

 protected:
  /**
   * @brief Data used by a casadi::Function object for evaluating the
   * casadi::Function object.
   *
   */
  struct CasadiFunctionData {
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
  };

  // Function
  mutable ::casadi::Function f_;
  mutable CasadiFunctionData f_data_;
  // Derivative
  mutable ::casadi::Function d_;
  mutable CasadiFunctionData d_data_;
  // Hessian
  mutable ::casadi::Function h_;
  mutable CasadiFunctionData h_data_;

 private:
  // Output matrix data
  std::vector<GenericEigenMatrix> f_out_;

  std::vector<GenericEigenMatrix> d_out_;

  std::vector<GenericEigenMatrix> h_out_;
};

}  // namespace casadi
}  // namespace damotion

#endif /* CASADI_FUNCTION_H */
