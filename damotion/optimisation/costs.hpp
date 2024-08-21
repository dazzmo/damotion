#ifndef COSTS_BASE_H
#define COSTS_BASE_H

#include <casadi/casadi.hpp>

#include "damotion/casadi/function.hpp"

namespace damotion {
namespace optimisation {

/**
 * @brief Generic cost with a single vector input
 *
 */
class Cost : public FunctionBase<1, double> {
 public:
  using SharedPtr = std::shared_ptr<Cost>;
  using UniquePtr = std::unique_ptr<Cost>;

  using Id = Index;

  using String = std::string;

  using Base = FunctionBase<1, double>;

  const String &name() const { return name_; }

  Cost(const String &name, const Index &nx, const Index &np = 0)
      : Base(), nx_(nx), np_(np) {}

  /**
   * @brief Size of the input vector for the cost \f$ c(x, p) \f$
   *
   * @return const Index&
   */
  const Index &nx() const { return nx_; }

  /**
   * @brief Size of the parameter vector for the cost \f$ c(x, p) \f$
   *
   * @return const Index&
   */
  const Index &np() const { return np_; }

  virtual double evaluate(const InputVectorType &x,
                          OptionalJacobianType grd = nullptr) const = 0;

 protected:
  void set_nx(const Index &nx) { nx_ = nx; }
  void set_np(const Index &np) { np_ = np; }

 private:
  Index nx_;
  Index np_;

  String name_ = "";

  /**
   * @brief Creates a unique id for each cost
   *
   * @return Id
   */
  Id createID() {
    static Id next_id = 0;
    Id id = next_id;
    next_id++;
    return id;
  }
};

/**
 * @brief Linear cost of the form \f$ c(x, p) = c^T(p) x + b(p) \in \mathbb{R}
 * \f$.
 *
 */
class LinearCost : public Cost {
 public:
  using UniquePtr = std::unique_ptr<LinearCost>;
  using SharedPtr = std::shared_ptr<LinearCost>;

  virtual void coeffs(OptionalVector c, double &b) const = 0;

  LinearCost(const String &name, const Index &nx, const Index &np = 0)
      : Cost(name, nx, np) {
    // Initialise coefficient matrices
    c_ = Eigen::VectorXd::Zero(this->nx());
    b_ = 0.0;
  }

  ReturnType evaluate(const InputVectorType &x,
                      OptionalJacobianType grd = nullptr) const {
    // Compute A and b
    coeffs(c_, b_);
    // Copy jacobian
    if (grd) grd = c_;
    // Compute linear constraint
    return c_.dot(x) + b_;
  }

 private:
  mutable Eigen::VectorXd c_;
  mutable double b_;
};

/**
 * @brief A cost of the form 0.5 x^T A x + b^T x + c
 *
 */
class QuadraticCost : public Cost {
 public:
  using UniquePtr = std::unique_ptr<QuadraticCost>;
  using SharedPtr = std::shared_ptr<QuadraticCost>;

  virtual void coeffs(OptionalHessianType A, OptionalVectorType b,
                      double &c) const = 0;

  QuadraticCost(const String &name, const Index &nx, const Index &np = 0)
      : Cost(name, nx, np) {
    A_ = HessianType::Zero(nx, nx);
    b_ = JacobianType::Zero(1, nx);
    c_ = 0.0;
  }

  ReturnType evaluate(const InputVectorType &x,
                      OptionalJacobianType g = nullptr) const {
    // Compute A and b
    coeffs(A_, b_, c_);
    // Copy jacobian
    if (g) *g = A_ * x + b_;
    // Compute linear cost
    return 0.5 * x.transpose() * A_ * x + b_.dot(x) + c_;
  }

 private:
  mutable HessianType A_;
  mutable VectorType b_;
  mutable double c_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* COSTS_BASE_H */
