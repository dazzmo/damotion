#ifndef CONSTRAINTS_LINEAR_H
#define CONSTRAINTS_LINEAR_H

#include "damotion/optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

template <typename MatrixType>
class LinearConstraint : public ConstraintBase<MatrixType> {
 public:
  using UniquePtr = std::unique_ptr<LinearConstraint<MatrixType>>;
  using SharedPtr = std::shared_ptr<LinearConstraint<MatrixType>>;

  /**
   * @brief Construct a new Linear Constraint object of the form \f$ A x + b
   * \f$
   *
   * @param name Name of the constraint. Default name given if provided ""
   * @param A Vector of coefficient matrices
   * @param b
   * @param p Parameters that A and b depend on
   * @param bounds
   * @param jac
   */
  LinearConstraint(const std::string &name, const casadi::SX &A,
                   const casadi::SX &b, const casadi::SXVector &p,
                   const BoundsType &bounds, bool jac = true)
      : ConstraintBase<MatrixType>(name, "linear_constraint") {
    ConstructConstraint(A, b, p, bounds, jac);
  }

  LinearConstraint(const std::string &name, const Eigen::MatrixXd &A,
                   const Eigen::VectorXd &b, const BoundsType &bounds,
                   bool jac = true)
      : ConstraintBase<MatrixType>(name, "linear_constraint") {
    // Constant vector b
    casadi::DM Ad, bd;
    damotion::casadi::toCasadi(b, bd);
    damotion::casadi::toCasadi(A, Ad);
    casadi::SX bsx = bd;
    casadi::SX Asx = Ad;

    // Construct constraint
    ConstructConstraint(Asx, bsx, {}, bounds, jac);
  }

  LinearConstraint(const std::string &name, const sym::Expression &ex,
                   const BoundsType &bounds, bool jac = true)
      : ConstraintBase<MatrixType>(name, "linear_constraint") {
    // Extract linear form
    casadi::SX A, b;
    casadi::SX::linear_coeff(ex, ex.Variables()[0], A, b, true);

    ConstructConstraint(A, b, ex.Parameters(), bounds, jac);
  }

  void SetCallback(
      const typename common::CallbackFunction<MatrixType>::f_callback_ &fA,
      const typename common::CallbackFunction<Eigen::VectorXd>::f_callback_
          &fb) {
    fA_ = std::make_shared<common::CallbackFunction<MatrixType>>(1, 1, fA);
    fb_ = std::make_shared<common::CallbackFunction<Eigen::VectorXd>>(1, 1, fb);
  }

  /**
   * @brief Returns the most recent evaluation of the constraint
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &Vector() const override { return c_; }
  /**
   * @brief The Jacobian of the constraint with respect to the i-th variable
   * vector
   *
   * @param i
   * @return const MatrixType&
   */
  const MatrixType &Jacobian() const override { return fA_->getOutput(0); }

  /**
   * @brief The coefficient matrix A for the expression A x + b.
   *
   * @return const MatrixType&
   */
  const Eigen::Ref<const MatrixType> A() const { return fA_->getOutput(0); }

  /**
   * @brief The constant vector for the linear constraint A x + b
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::Ref<const Eigen::VectorXd> b() const {
    return fb_->getOutput(0);
  }

  /**
   * @brief Evaluate the constraint and Jacobian (optional) given input
   * variables x and parameters p.
   *
   * @param x
   * @param p
   * @param jac Flag for computing the Jacobian
   */
  void eval(const common::InputRefVector &x, const common::InputRefVector &p,
            bool jac = true) const override {
    VLOG(10) << this->name() << " eval()";
    // Evaluate the coefficients
    fA_->call(p);
    fb_->call(p);
    VLOG(10) << "A";
    VLOG(10) << fA_->getOutput(0);
    VLOG(10) << "b";
    VLOG(10) << fb_->getOutput(0);
    VLOG(10) << "x";
    VLOG(10) << x[0];
    // Evaluate the constraint
    c_ = fA_->getOutput(0) * x[0] + fb_->getOutput(0);
  }

  void eval_hessian(const common::InputRefVector &x, const Eigen::VectorXd &l,
                    const common::InputRefVector &p) const override {
    // No need, as linear constraints do not have a Hessian
  }

 private:
  std::shared_ptr<common::Function<MatrixType>> fA_;
  std::shared_ptr<common::Function<Eigen::VectorXd>> fb_;

  // Contraint vector (c = Ax + b)
  mutable Eigen::VectorXd c_;

  void ConstructConstraint(const casadi::SX &A, const casadi::SX &b,
                           const casadi::SXVector &p, const BoundsType &bounds,
                           bool jac = true, bool sparse = false) {
    assert(A.rows() == b.rows() && "A and b must be same dimension!");
    casadi::SXVector in = {};

    VLOG(10) << this->name() << " ConstructConstraint()";
    VLOG(10) << "A = " << A;
    VLOG(10) << "b = " << b;

    // Create constraint dimensions and update bounds
    this->Resize(b.rows(), in.size(), p.size());
    this->SetBounds(bounds);

    // Add any parameters that define A and b
    for (const casadi::SX &pi : p) {
      in.push_back(pi);
    }

    if (std::is_same<MatrixType, Eigen::SparseMatrix<double>>::value) {
      fA_ = std::make_shared<damotion::casadi::FunctionWrapper<MatrixType>>(
          casadi::Function(this->name() + "_A", in, {A}));
    } else {
      fA_ = std::make_shared<damotion::casadi::FunctionWrapper<MatrixType>>(
          casadi::Function(this->name() + "_A", in, {densify(A)}));
    }

    fb_ = std::make_shared<damotion::casadi::FunctionWrapper<Eigen::VectorXd>>(
        casadi::Function(this->name() + "_b", in, {densify(b)}));

    this->has_jac_ = true;
    this->has_hes_ = false;
  }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_LINEAR_H */
