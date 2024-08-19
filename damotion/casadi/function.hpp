#ifndef DAMOTION_CASADI_FUNCTION_HPP
#define DAMOTION_CASADI_FUNCTION_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pinocchio/autodiff/casadi.hpp>

#include "damotion/core/function.hpp"
#include "damotion/optimisation/constraints/constraints.h"
#include "damotion/optimisation/costs/costs.h"

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
            alwaysT<Os, OptionalMatrix> ...out) {
    std::size_t cnt = 0;
    for (const Eigen::VectorXd &xi : {in...}) {
      in_data_ptr_[cnt++] = xi.data();
    }

    cnt = 0;
    for (OptionalMatrix oi : {out...}) {
      if (oi) {
        out_data_ptr_[cnt++] = oi->data();
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

// template <typename Seq>
// class computeJacobianImpl;

// // Templated functions to compute jacobians and hessians
// template <class... InputSequence>
// ::casadi::Function
// computeJacobiansImpl<std::index_sequence<InputSequence...>>(
//     const ::casadi::SX &ex, const alwaysT<InputSequence, ::casadi::SX> &...x,
//     const ::casadi::SX &p = ::casadi::SX()) {
//   std::vector<SymbolicVector> xv = {};
//   // Add variables
//   for (const SymbolicVector &xi : {x...}) {
//     xv.push_back(xi);
//   }
//   // Add parameters
//   if (!p.isNull()) {
//     xv.push_back(p);
//   }

//   // Create function for ex
//   ::casadi::Function f("ex", {ex}, xv);
//   f_ = std::make_unique<FunctionWrapper<N>>(f);

//   std::size_t cnt = 0;
//   std::vector<SymbolicExpression> grd = {};

//   for (const SymbolicVector &xi : {x...}) {
//     // Compute jacobian and hessian, if applicable
//     grd.push_back(::casadi::SX::gradient(ex, xi));
//   }
//   // Create function
//   fjac_.push_back(std::make_unique<FunctionWrapper<N>>(
//       ::casadi::Function("fgrd", grd, xv)));

//   return ::casadi::Function();
// }

// // template<std::size_t N>
// // using computeJacobians()

// // ::casadi::Function computeHessians() {}

// // Constraint construction through Casadi

class LinearConstraint : public optimisation::LinearConstraint {
 public:
  // Override method
  LinearConstraint(const ::casadi::SX &ex, const ::casadi::SX &x,
                   const ::casadi::SX &p = ::casadi::SX())
      : optimisation::LinearConstraint(""), coeffs_(nullptr) {
    // Create coefficients
    ::casadi::SX A, b;
    ::casadi::SX::linear_coeff(ex, x, A, b, true);
    // Create function
    coeffs_ = std::make_unique<FunctionWrapper<1, 2>>(
        ::casadi::Function("", {A, b}, {p}));
  }

  void coeffs(OptionalJacobianType A = nullptr,
              OptionalVectorType b = nullptr) const override {
    OptionalMatrix btmp(b);
    coeffs_->call(get_parameters(), A, btmp);
  }

 private:
  JacobianType A_;
  InputVectorType b_;

  FunctionWrapper<1, 2>::UniquePtr coeffs_;
};

// class LinearCost : public optimisation::LinearCost {
//  public:
//   // Override method
//   LinearCost(const ::casadi::SX &ex, const ::casadi::SX &x,
//              const ::casadi::SX &p = ::casadi::SX())
//       : coeffs_(nullptr) {
//     // Create coefficients
//     ::casadi::SX A, b;
//     ::casadi::SX::linear_coeffs(ex, x, A, b);
//     // Create function
//     coeffs_ =
//         std::make_unique<FunctionWrapper>(::casadi::Function("", {A, b},
//         {p}));
//   }

//   void coeffs(OptionalJacobianType A = nullptr,
//               OptionalVector b = nullptr) override {
//     coeffs_->call(get_parameters(), {A, b});
//   }

//  private:
//   FunctionWrapper::UniquePtr coeffs_;
// };

// class QuadraticCost : public optimisation::QuadraticCost {
//  public:
//   // Override method
//   QuadraticCost(const ::casadi::SX &ex, const ::casadi::SX &x,
//                 const ::casadi::SX &p = ::casadi::SX())
//       : coeffs_(nullptr) {
//     // Create coefficients
//     ::casadi::SX A, b, c;
//     ::casadi::SX::quadratic_coeffs(ex, x, A, b, c);
//     // Create function
//     coeffs_ = std::make_unique<FunctionWrapper>(
//         ::casadi::Function("", {A, b, c}, {p}));
//   }

//   void coeffs(OptionalHessianType A = nullptr, OptionalJacobianType b =
//   nullptr,
//               const double &c = 0.0) override {
//     coeffs_->call(get_parameters(), {A, b, c});
//   }

//  private:
//   FunctionWrapper::UniquePtr coeffs_;
// };

}  // namespace casadi
}  // namespace damotion

#endif/* DAMOTION_CASADI_FUNCTION_HPP */
