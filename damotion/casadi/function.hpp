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

template <std::size_t InputSize, std::size_t OutputSize>
class FunctionWrapper {
 public:
  using UniquePtr = std::unique_ptr<FunctionWrapper>;
  using SharedPtr = std::shared_ptr<FunctionWrapper>;

  FunctionWrapper() = default;
  FunctionWrapper(const ::casadi::Function &f) { *this = f; }

  ~FunctionWrapper() {
    // Release memory for casadi function
    if (!f_.is_null()) f_.release(mem_);
  }

  FunctionWrapper<InputSize, OutputSize> &operator=(
      const FunctionWrapper &other) {
    *this = other.f_;
    return *this;
  }

  FunctionWrapper<InputSize, OutputSize> &operator=(::casadi::Function f) {
    if (f.is_null()) {
      return *this;
    }

    f_ = f;

    // Checkout memory object for function
    mem_ = f_.checkout();

    // Resize work vectors
    in_data_ptr_.assign(f_.n_in(), nullptr);
    out_data_ptr_.assign(f_.n_out(), nullptr);

    iw_.assign(f_.sz_iw(), 0);
    dw_.assign(f_.sz_w(), 0.0);

    return *this;
  }

  void call(const std::array<Eigen::Ref<const Eigen::VectorXd>, InputSize> &in,
            const std::array<OptionalMatrix, OutputSize> &out) {
    for (std::size_t i = 0; i < InputSize; ++i) {
      in_data_ptr_[i] = in[i].data();
    }

    for (std::size_t i = 0; i < OutputSize; ++i) {
      if (out[i]) {
        out_data_ptr_[i] = const_cast<double *>(out[i]->data());
      } else {
        out_data_ptr_[i] = nullptr;
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

/**
 * @brief Generic constraint function
 *
 */
class Constraint : public optimisation::Constraint {
 public:
  Constraint(const String &name, const ::casadi::SX &ex, const ::casadi::SX &x,
             const ::casadi::SX &p = ::casadi::SX())
      : optimisation::Constraint(name, ex.size1(), x.size1()),
        function_(nullptr),
        jacobian_(nullptr),
        hessian_(nullptr) {
    // Compute Jacobian
    ::casadi::SX jac = ::casadi::SX::jacobian(ex, x);

    // Function
    function_ = std::make_unique<FunctionWrapper<2, 1>>(
        ::casadi::Function("function", {x, p}, {densify(ex)}));
    // Jacobian
    jacobian_ = std::make_unique<FunctionWrapper<2, 1>>(
        ::casadi::Function("jacobian", {x, p}, {densify(jac)}));

    // Hessian
    ::casadi::SX l = ::casadi::SX::sym("l", ex.size1());
    ::casadi::SX lc = ::casadi::SX::mtimes(l.T(), ex);
    ::casadi::SX hes = ::casadi::SX::hessian(lc, x);

    hessian_ = std::make_unique<FunctionWrapper<3, 1>>(
        ::casadi::Function("hessian", {x, l, p}, {densify(hes)}));
  }

  Eigen::VectorXd evaluate(const InputVectorType &x,
                           OptionalJacobianType jac = nullptr) const override {
    VectorType out(this->size());
    function_->call({x, get_parameters()}, {out});
    if (jac) jacobian_->call({x, get_parameters()}, {jac});
    return out;
  }

  void hessian(const InputVectorType &x, const ReturnType &lam,
               OptionalHessianType hes = nullptr) const override {
    if (hes) {
      hessian_->call({x, lam, get_parameters()}, {hes});
    }
  }

 private:
  FunctionWrapper<2, 1>::UniquePtr function_;
  FunctionWrapper<2, 1>::UniquePtr jacobian_;
  FunctionWrapper<3, 1>::UniquePtr hessian_;
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
    coeffs_->call({get_parameters()}, {A, b});
  }

 private:
  FunctionWrapper<1, 2>::UniquePtr coeffs_;
};

class Cost : public optimisation::Cost {
 public:
  // Override method
  Cost(const String &name, const ::casadi::SX &ex, const ::casadi::SX &x,
       const ::casadi::SX &p = ::casadi::SX())
      : optimisation::Cost(name, ex.size1(), x.size1()),
        function_(nullptr),
        gradient_(nullptr),
        hessian_(nullptr) {
    // Compute Jacobian
    ::casadi::SX grd = ::casadi::SX::jacobian(ex, x);
    // Compute Hessian
    ::casadi::SX hes = ::casadi::SX::hessian(ex, x);

    // Create function
    function_ = std::make_unique<FunctionWrapper<2, 1>>(
        ::casadi::Function("function", {x, p}, {densify(ex)}));
    gradient_ = std::make_unique<FunctionWrapper<2, 1>>(
        ::casadi::Function("jacobian", {x, p}, {densify(grd)}));
    hessian_ = std::make_unique<FunctionWrapper<2, 1>>(
        ::casadi::Function("hessian", {x, p}, {densify(hes)}));
  }

  ReturnType evaluate(const InputVectorType &x,
                      OptionalJacobianType grd = nullptr) const override {
    ReturnType out;
    function_->call({x, get_parameters()}, {out});
    if (grd) gradient_->call({x, get_parameters()}, {grd});
    return out;
  }

  void hessian(const InputVectorType &x, const double &lam = 1.0,
               OptionalHessianType hes = nullptr) const override {
    if (hes) {
      hessian_->call({x, get_parameters()}, {hes});
      (*hes) *= lam;
    }
  }

 private:
  FunctionWrapper<2, 1>::UniquePtr function_;
  FunctionWrapper<2, 1>::UniquePtr gradient_;
  FunctionWrapper<2, 1>::UniquePtr hessian_;
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
    coeffs_->call({get_parameters()}, {c, b});
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
    coeffs_->call({get_parameters()}, {A, b, c});
  }

 private:
  FunctionWrapper<1, 3>::UniquePtr coeffs_;
};

}  // namespace casadi
}  // namespace damotion

#endif /* DAMOTION_CASADI_FUNCTION_HPP */
