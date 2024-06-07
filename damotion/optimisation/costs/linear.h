#ifndef COSTS_LINEAR_H
#define COSTS_LINEAR_H

#include "damotion/optimisation/costs/base.h"

namespace damotion {
namespace optimisation {

/**
 * @brief Cost of the form \f$ c^T x + b \f$
 *
 */
class LinearCost : public CostBase {
 public:
  LinearCost(const std::string &name, const Eigen::VectorXd &c, const double &b,
             bool jac = true) {
    // Create Costs
    casadi::DM cd, bd = b;
    damotion::casadi::toCasadi(c, cd);
    casadi::SX cs = cd, bs = bd;

    ConstructCost(name, cs, bs, {}, jac, true);
  }

  LinearCost(const std::string &name, const casadi::SX &c, const casadi::SX &b,
             const casadi::SXVector &p, bool jac = true) {
    ConstructCost(name, c, b, p, jac, true);
  }

  LinearCost(const std::string &name, const casadi::SX &ex,
             const casadi::SXVector &x, const casadi::SXVector &p,
             bool jac = true, bool hes = true) {
    int nvar = 0;
    casadi::SXVector in = {};
    // Extract quadratic form
    casadi::SX c, b;
    casadi::SX::linear_coeff(ex, x[0], c, b, true);

    ConstructCost(name, c, b, p, jac, hes);
  }

  /**
   * @brief Returns the coefficients of x for the cost expression.
   *
   * @return Eigen::VectorXd
   */
  const GenericEigenMatrix &c() { return fc_->getOutput(0); }

  /**
   * @brief Returns the constant term b in the cost expression.
   *
   * @return const double
   */
  const GenericEigenMatrix &b() { return fb_->getOutput(0); }

 private:
  common::Function::SharedPtr fc_;
  common::Function::SharedPtr fb_;

  void ConstructCost(const std::string &name, const casadi::SX &c,
                     const casadi::SX &b, const casadi::SXVector &p,
                     bool jac = true, bool hes = true, bool sparse = false) {
    this->SetName(name);

    int nvar = 0;
    casadi::SXVector in = {};
    for (const casadi::SX &pi : p) {
      in.push_back(pi);
    }

    casadi::SX c_tmp = c, b_tmp = b;
    if (!sparse) {
      c_tmp = densify(c_tmp);
      b_tmp = densify(b_tmp);
    }

    // Create coefficient functions
    fc_ = std::make_shared<damotion::casadi::FunctionWrapper>(
        casadi::Function(this->name() + "_A", in, {c_tmp}));
    fb_ = std::make_shared<damotion::casadi::FunctionWrapper>(
        casadi::Function(this->name() + "_b", in, {b_tmp}));

    this->has_grd_ = true;
    // Indicate cost does not have a hessian
    this->has_hes_ = false;
  }
};

}  // namespace optimisation

}  // namespace damotion

#endif /* COSTS_LINEAR_H */
