#ifndef DAMOTION_CASADI_FUNCTION_HPP
#define DAMOTION_CASADI_FUNCTION_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pinocchio/autodiff/casadi.hpp>

#include "damotion/core/function.hpp"
#include "damotion/optimisation/constraints.hpp"
#include "damotion/optimisation/costs.hpp"

namespace damotion {
namespace casadi {

template <typename InSeq, typename OutSeq>
class FunctionWrapperImpl;

template <std::size_t... Is, std::size_t... Os>
class FunctionWrapperImpl<std::index_sequence<Is...>,
                          std::index_sequence<Os...>> {
 public:
  using UniquePtr = std::unique_ptr<FunctionWrapperImpl>;
  using SharedPtr = std::shared_ptr<FunctionWrapperImpl>;

  FunctionWrapperImpl() = default;
  FunctionWrapperImpl(const ::casadi::Function &f) { *this = f; }

  ~FunctionWrapperImpl() {
    // Release memory for casadi function
    if (!f_.is_null()) f_.release(mem_);
  }

  FunctionWrapperImpl<std::index_sequence<Is...>, std::index_sequence<Os...>> &
  operator=(const FunctionWrapperImpl &other) {
    *this = other.f_;
    return *this;
  }

  FunctionWrapperImpl<std::index_sequence<Is...>, std::index_sequence<Os...>> &
  operator=(::casadi::Function f) {
    if (f.is_null()) {
      return *this;
    }

    f_ = f;

    // Checkout memory object for function
    mem_ = f.checkout();

    // Resize work vectors
    in_data_ptr_.assign(f_.sz_arg(), nullptr);
    out_data_ptr_.assign(f_.n_out(), nullptr);

    iw_.assign(f_.sz_iw(), 0);
    dw_.assign(f_.sz_w(), 0.0);

    return *this;
  }

  void call(const alwaysT<Is, Eigen::VectorXd> &...in,
            alwaysT<Os, OptionalMatrix>... out) {
    std::size_t cnt = 0;
    for (const Eigen::VectorXd &xi : {in...}) {
      in_data_ptr_[cnt++] = xi.data();
    }

    cnt = 0;
    for (const OptionalMatrix &oi : {out...}) {
      if (oi) {
        out_data_ptr_[cnt++] = const_cast<double *>(oi->data());
      } else {
        out_data_ptr_[cnt++] = nullptr;
      }
    }
    // Call the function
    f_(in_data_ptr_.data(), out_data_ptr_.data(), iw_.data(), dw_.data(), mem_);
  }

 private:
  // Data input vector for casadi function
  std::vector<const double *> in_data_ptr_;
  // Data output pointers for casadi function
  std::vector<double *> out_data_ptr_;

  // Memory allocated for function evaluation
  int mem_;

  // Integer working vector
  std::vector<casadi_int> iw_;
  // Double working vector
  std::vector<double> dw_;

  // Underlying function
  ::casadi::Function f_;
};

template <std::size_t InputSize, std::size_t OutputSize>
using FunctionWrapper =
    FunctionWrapperImpl<std::make_index_sequence<InputSize>,
                        std::make_index_sequence<OutputSize>>;

/**
 * @brief Generic constraint function
 *
 */
class Constraint : public optimisation::Constraint {
 public:
  // Override method
  Constraint(const String &name, const ::casadi::SX &ex, const ::casadi::SX &x,
             const ::casadi::SX &p = ::casadi::SX())
      : optimisation::Constraint(name, ex.size1(), x.size1()),
        function_(nullptr),
        jacobian_(nullptr) {
    // Compute Jacobian
    ::casadi::SX jac = ::casadi::SX::jacobian(ex, x);

    // Create function
    function_ = std::make_unique<FunctionWrapper<2, 1>>(
        ::casadi::Function("function", {x, p}, {densify(ex)}));
    jacobian_ = std::make_unique<FunctionWrapper<2, 1>>(
        ::casadi::Function("jacobian", {x, p}, {densify(jac)}));

    // TODO - Hessian calculations
  }

  Eigen::VectorXd evaluate(const InputVectorType &x,
                           OptionalJacobianType jac = nullptr) const override {
    VectorType out(this->size());
    function_->call(x, get_parameters(), out);
    if (jac) jacobian_->call(x, get_parameters(), jac);
    return out;
  }

 private:
  FunctionWrapper<2, 1>::UniquePtr function_;
  FunctionWrapper<2, 1>::UniquePtr jacobian_;
};

class LinearConstraint : public optimisation::LinearConstraint {
 public:
  // Override method
  LinearConstraint(const String &name, const ::casadi::SX &ex,
                   const ::casadi::SX &x,
                   const ::casadi::SX &p = ::casadi::SX())
      : optimisation::LinearConstraint(name, ex.size1(), x.size1()),
        coeffs_(nullptr) {
    // Create coefficients
    ::casadi::SX A, b;
    ::casadi::SX::linear_coeff(ex, x, A, b, true);
    // Create function
    coeffs_ = std::make_unique<FunctionWrapper<1, 2>>(
        ::casadi::Function("linear_coeffs", {p}, {densify(A), densify(b)}));
  }

  void coeffs(OptionalJacobianType A = nullptr,
              OptionalVectorType b = nullptr) const override {
    coeffs_->call(get_parameters(), A, b);
  }

 private:
  FunctionWrapper<1, 2>::UniquePtr coeffs_;
};

class LinearCost : public optimisation::LinearCost {
 public:
  // Override method
  LinearCost(const String &name, const ::casadi::SX &ex, const ::casadi::SX &x,
             const ::casadi::SX &p = ::casadi::SX())
      : optimisation::LinearCost(name, x.size1(), p.size1()), coeffs_(nullptr) {
    // Create coefficients
    ::casadi::SX c, b;
    ::casadi::SX::linear_coeff(ex, x, c, b, true);
    // Create function
    coeffs_ = std::make_unique<FunctionWrapper<1, 2>>(
        ::casadi::Function("linear_coeffs", {p}, {densify(c), densify(b)}));
  }

  void coeffs(OptionalVector c, double &b) const override {
    coeffs_->call(get_parameters(), c, b);
  }

 private:
  FunctionWrapper<1, 2>::UniquePtr coeffs_;
};

class QuadraticCost : public optimisation::QuadraticCost {
 public:
  // Override method
  QuadraticCost(const String &name, const ::casadi::SX &ex,
                const ::casadi::SX &x, const ::casadi::SX &p = ::casadi::SX())
      : optimisation::QuadraticCost(name, x.size1(), p.size1()),
        coeffs_(nullptr) {
    // Create coefficients
    ::casadi::SX A, b, c;
    ::casadi::SX::quadratic_coeff(ex, x, A, b, c, true);
    // Create function
    // todo - check compile time that these number of outputs are correct for
    // todo - the function
    coeffs_ = std::make_unique<FunctionWrapper<1, 3>>(
        ::casadi::Function("quadratic_coeffs", {p},
                           {::casadi::SX::densify(A), ::casadi::SX::densify(b),
                            ::casadi::SX::densify(c)}));
  }

  void coeffs(OptionalHessianType A, OptionalVectorType b,
              double &c) const override {
    coeffs_->call(get_parameters(), A, b, c);
  }

 private:
  FunctionWrapper<1, 3>::UniquePtr coeffs_;
};

}  // namespace casadi
}  // namespace damotion

#endif /* DAMOTION_CASADI_FUNCTION_HPP */
