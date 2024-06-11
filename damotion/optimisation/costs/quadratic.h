#ifndef COSTS_QUADRATIC_H
#define COSTS_QUADRATIC_H
#include <Eigen/Core>

#include "damotion/optimisation/costs/base.h"

namespace damotion {
namespace optimisation {
/**
 * @brief A cost of the form 0.5 x^T Q x + g^T x + c
 *
 */
class QuadraticCost : public Cost {
 public:
  using UniquePtr = std::unique_ptr<QuadraticCost>;
  using SharedPtr = std::shared_ptr<QuadraticCost>;

  QuadraticCost(const std::string &name, const Eigen::MatrixXd &A,
                const Eigen::VectorXd &b, const double &c, bool jac = true,
                bool hes = true) {
    // Cost
    casadi::DM Ad, bd, cd = c;
    damotion::casadi::toCasadi(A, Ad);
    damotion::casadi::toCasadi(b, bd);
    ConstructConstraint(name, Ad, bd, cd, {}, jac, hes);
  }

  QuadraticCost(const std::string &name, const casadi::SX &A,
                const casadi::SX &b, const casadi::SX &c,
                const casadi::SXVector &p, bool jac = true, bool hes = true) {
    ConstructConstraint(name, A, b, c, p, jac, hes);
  }

  QuadraticCost(const std::string &name, const casadi::SX &ex,
                const casadi::SX &x, const casadi::SXVector &p, bool jac = true,
                bool hes = true) {
    // Extract quadratic form
    casadi::SX A, b, c;
    casadi::SX::quadratic_coeff(ex, x, A, b, c, true);
    // Remove factor of two from hessian
    A *= 0.5;
    ConstructConstraint(name, A, b, c, p, jac, hes);
  }

  /**
   * @brief Lower triangle representation of the quadratic cost Hessian
   *
   * @return const MatrixType&
   */
  const GenericEigenMatrix &A() const {
    fA_->call();
    return fA_->GetOutput(0);
  }

  const GenericEigenMatrix &b() const {
    fb_->call();
    return fb_->GetOutput(0);
  }

  const GenericEigenMatrix &c() const {
    fc_->call();
    return fc_->GetOutput(0);
  }

 private:
  common::Function::UniquePtr fA_;
  common::Function::UniquePtr fb_;
  common::Function::UniquePtr fc_;

  void ConstructConstraint(const std::string &name, const casadi::SX &A,
                           const casadi::SX &b, const casadi::SX &c,
                           const casadi::SXVector &p, bool jac = true,
                           bool hes = true, bool sparse = false) {
    this->SetName(name);

    // Create expression
    int nx = A.columns();

    // Quadratic cost
    casadi::SX x = casadi::SX::sym("x", nx);
    casadi::SX ex = mtimes(x.T(), mtimes(A, x)) + mtimes(b.T(), x) + c;

    GenerateFunction({ex}, {x}, p, jac, hes, sparse);

    // Create specialised functions for A, b and c
    fA_ = std::make_unique<damotion::casadi::CasadiFunction>(
        ::casadi::SXVector({A}), ::casadi::SXVector(), p, false, false, sparse);
    fb_ = std::make_unique<damotion::casadi::CasadiFunction>(
        ::casadi::SXVector({b}), ::casadi::SXVector(), p, false, false, sparse);
    fc_ = std::make_unique<damotion::casadi::CasadiFunction>(
        ::casadi::SXVector({c}), ::casadi::SXVector(), p, false, false, sparse);
  }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* COSTS_QUADRATIC_H */
