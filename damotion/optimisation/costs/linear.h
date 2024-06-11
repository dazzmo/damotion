#ifndef COSTS_LINEAR_H
#define COSTS_LINEAR_H

#include "damotion/optimisation/costs/base.h"

namespace damotion {
namespace optimisation {

/**
 * @brief Cost of the form \f$ c^T x + b \f$
 *
 */
class LinearCost : public Cost {
 public:
  LinearCost(const std::string &name, const Eigen::VectorXd &c, const double &b,
             bool jac = true) {
    // Create Costs
    casadi::DM cd, bd = b;
    damotion::casadi::toCasadi(c, cd);
    ConstructCost(name, cd, bd, {}, jac, true);
  }

  LinearCost(const std::string &name, const casadi::SX &c, const casadi::SX &b,
             const casadi::SXVector &p, bool jac = true) {
    ConstructCost(name, c, b, p, jac, true);
  }

  LinearCost(const std::string &name, const casadi::SX &ex, const casadi::SX &x,
             const casadi::SXVector &p, bool jac = true, bool hes = true) {
    // Extract quadratic form
    casadi::SX c, b;
    casadi::SX::linear_coeff(ex, x, c, b, true);
    ConstructCost(name, c, b, p, jac, hes);
  }

  /**
   * @brief Returns the coefficients of x for the cost expression.
   *
   * @return Eigen::VectorXd
   */
  const GenericEigenMatrix &c() {
    fc_->call();
    return fc_->GetOutput(0);
  }

  /**
   * @brief Returns the constant term b in the cost expression.
   *
   * @return const double
   */
  const GenericEigenMatrix &b() {
    fb_->call();
    return fb_->GetOutput(0);
  }

 private:
  common::Function::UniquePtr fc_;
  common::Function::UniquePtr fb_;

  void ConstructCost(const std::string &name, const casadi::SX &c,
                     const casadi::SX &b, const casadi::SXVector &p,
                     bool jac = true, bool hes = true, bool sparse = false) {
    assert(b.rows() == 1 && "b must be scalar!");

    this->SetName(name);

    // Create expression
    int nx = c.rows();

    ::casadi::SX x = ::casadi::SX::sym("x", nx);
    ::casadi::SX ex = mtimes(c.T(), x) + b;

    GenerateFunction({ex}, {x}, p, jac, false, sparse);

    // Create coefficient functions
    fc_ = std::make_unique<damotion::casadi::CasadiFunction>(
        ::casadi::SXVector({c}), ::casadi::SXVector(), p, false, false, sparse);
    fb_ = std::make_unique<damotion::casadi::CasadiFunction>(
        ::casadi::SXVector({b}), ::casadi::SXVector(), p, false, false, sparse);
  }
};

}  // namespace optimisation

}  // namespace damotion

#endif /* COSTS_LINEAR_H */
