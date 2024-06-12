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
  LinearCost(const std::string &name, const casadi::SX &c, const casadi::SX &b,
             const casadi::SX &x, const casadi::SXVector &p)
      : Cost(mtimes(c.T(), x) + b, {x}, p, false) {
    // Create specialty functions for coefficient evaluation
    ::casadi::Function fc("c", p, {c}), fb("b", p, {b});
    fc_ = std::make_shared<damotion::casadi::FunctionWrapper>(fc);
    fb_ = std::make_shared<damotion::casadi::FunctionWrapper>(fb);
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
  common::Function::SharedPtr fc_;
  common::Function::SharedPtr fb_;
};

}  // namespace optimisation

}  // namespace damotion

#endif /* COSTS_LINEAR_H */
