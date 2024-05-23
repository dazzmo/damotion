#ifndef CONSTRAINTS_LINEAR_H
#define CONSTRAINTS_LINEAR_H

#include "damotion/optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

class LinearConstraint : public ConstraintBase {
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
    casadi::SX bsx = bd;
    casadi::SX Asx = Ad;

    // Construct constraint
    ConstructConstraint(name, Asx, bsx, {}, bounds, jac);
  }

  LinearConstraint(const std::string &name, const casadi::SX &ex,
                   const casadi::SXVector &x, const casadi::SXVector &p,
                   const BoundsType &bounds, bool jac = true) {
    // Extract linear form
    casadi::SX A, b;
    casadi::SX::linear_coeff(ex, x[0], A, b, true);

    ConstructConstraint(name, A, b, p, bounds, jac);
  }

  void SetCallback(const typename common::CallbackFunction::f_callback_ &fA,
                   const typename common::CallbackFunction::f_callback_ &fb) {
    fA_ = std::make_shared<common::CallbackFunction>(1, 1, fA);
    fb_ = std::make_shared<common::CallbackFunction>(1, 1, fb);
  }

  /**
   * @brief The coefficient matrix A for the expression A x + b.
   *
   * @return const GenericMatrixData&
   */
  const GenericMatrixData &A() const { return fA_->getOutput(0); }

  /**
   * @brief The constant vector for the linear constraint A x + b
   *
   * @return const GenericMatrixData&
   */
  const GenericMatrixData &b() const { return fb_->getOutput(0); }

 private:
  common::Function::SharedPtr fA_;
  common::Function::SharedPtr fb_;

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

    // Create constraint dimensions and update bounds
    this->Resize(b.rows(), in.size(), p.size());
    this->SetBounds(bounds);

    // Add any parameters that define A and b
    for (const casadi::SX &pi : p) {
      in.push_back(pi);
    }

    casadi::SX A_tmp = A, b_tmp = b;
    if (!sparse) {
      A_tmp = densify(A_tmp);
      b_tmp = densify(b_tmp);
    }

    fA_ = std::make_shared<damotion::casadi::FunctionWrapper>(
        casadi::Function(this->name() + "_A", in, {A_tmp}));

    fb_ = std::make_shared<damotion::casadi::FunctionWrapper>(
        casadi::Function(this->name() + "_b", in, {b_tmp}));

    this->has_jac_ = true;
    this->has_hes_ = false;
  }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_LINEAR_H */
