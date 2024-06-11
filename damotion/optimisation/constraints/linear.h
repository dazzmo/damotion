#ifndef CONSTRAINTS_LINEAR_H
#define CONSTRAINTS_LINEAR_H

#include "damotion/optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

class LinearConstraint : public Constraint {
 public:
  using UniquePtr = std::unique_ptr<LinearConstraint>;
  using SharedPtr = std::shared_ptr<LinearConstraint>;

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
                   const BoundsType &bounds, bool jac = true) {
    ConstructConstraint(name, A, b, p, bounds, jac);
  }

  LinearConstraint(const std::string &name, const Eigen::MatrixXd &A,
                   const Eigen::VectorXd &b, const BoundsType &bounds,
                   bool jac = true) {
    // Constant vector b
    casadi::DM Ad, bd;
    damotion::casadi::toCasadi(b, bd);
    damotion::casadi::toCasadi(A, Ad);

    // Construct constraint
    ConstructConstraint(name, Ad, bd, {}, bounds, jac);
  }

  LinearConstraint(const std::string &name, const casadi::SX &ex,
                   const casadi::SXVector &x, const casadi::SXVector &p,
                   const BoundsType &bounds, bool jac = true) {
    // Extract linear form
    casadi::SX A, b;
    casadi::SX::linear_coeff(ex, x[0], A, b, true);

    ConstructConstraint(name, A, b, p, bounds, jac);
  }

  /**
   * @brief The coefficient matrix A for the expression A x + b.
   *
   * @return const GenericEigenMatrix&
   */
  const GenericEigenMatrix &A() const {
    // TODO - Set inputs
    fA_->call();
    return fA_->GetOutput(0);
  }

  /**
   * @brief The constant vector for the linear constraint A x + b
   *
   * @return const GenericEigenMatrix&
   */
  const GenericEigenMatrix &b() const {
    fb_->call();
    return fb_->GetOutput(0);
  }

 private:
  mutable CasadiFunction::UniquePtr fA_;
  mutable CasadiFunction::UniquePtr fb_;

  void ConstructConstraint(const std::string &name, const casadi::SX &A,
                           const casadi::SX &b, const casadi::SXVector &p,
                           const BoundsType &bounds, bool jac = true,
                           bool sparse = false) {
    assert(A.rows() == b.rows() && "A and b must be same dimension!");
    this->SetName(name);

    casadi::SXVector in = {};

    VLOG(10) << this->name() << " ConstructConstraint()";
    VLOG(10) << "A = " << A;
    VLOG(10) << "b = " << b;

    this->SetBounds(bounds);
    // Create specialised functions for A and b
    fA_ = std::make_unique<damotion::casadi::CasadiFunction>(
        ::casadi::SXVector({A}), ::casadi::SXVector(), p, false, false, sparse);
    fb_ = std::make_unique<damotion::casadi::CasadiFunction>(
        ::casadi::SXVector({b}), ::casadi::SXVector(), p, false, false, sparse);
  }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_LINEAR_H */
