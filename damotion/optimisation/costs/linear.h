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
             bool jac = true)
      : CostBase(name, "linear_cost") {
    // Create Costs
    casadi::DM cd, bd = b;
    damotion::casadi::toCasadi(c, cd);
    casadi::SX cs = cd, bs = bd;

    ConstructCost(cs, bs, {}, jac, true);
  }

  LinearCost(const std::string &name, const casadi::SX &c, const casadi::SX &b,
             const casadi::SXVector &p, bool jac = true) {
    ConstructCost(c, b, p, jac, true);
  }

  LinearCost(const std::string &name, const casadi::SX &ex,
             const casadi::SXVector &x, const casadi::SXVector &p,
             bool jac = true, bool hes = true) {
    int nvar = 0;
    casadi::SXVector in = {};
    // Extract quadratic form
    casadi::SX c, b;
    casadi::SX::linear_coeff(ex, x[0], c, b, true);

    ConstructCost(c, b, p, jac, hes);
  }

  /**
   * @brief Evaluate the constraint and gradient (optional) given input
   * variables x and parameters p.
   *
   * @param x
   * @param p
   * @param jac Flag for computing the Jacobian
   */
  void eval(const common::InputRefVector &x, const common::InputRefVector &p,
            bool grd = true) const override {
    VLOG(10) << this->name() << " eval()";
    // Evaluate the coefficients
    fc_->call(p);
    fb_->call(p);
    // Evaluate the constraint
    this->obj_ = c().dot(x[0]) + b();
    this->grd_ = c();
  }

  void eval_hessian(const common::InputRefVector &x,
                    const common::InputRefVector &p) const override {
    // No need, as linear costs do not have a Hessian
  }

  /**
   * @brief Returns the coefficients of x for the cost expression.
   *
   * @return Eigen::VectorXd
   */
  const GenericMatrixData &c() { return fc_->getOutput(0); }

  /**
   * @brief Returns the constant term b in the cost expression.
   *
   * @return const double
   */
  const double &b() { return fb_->getOutput(0); }

 private:
  std::shared_ptr<common::Function<Eigen::VectorXd>> fc_;
  std::shared_ptr<common::Function<double>> fb_;

  void ConstructCost(const casadi::SX &c, const casadi::SX &b,
                     const casadi::SXVector &p, bool jac = true,
                     bool hes = true) {
    int nvar = 0;
    casadi::SXVector in = {};
    for (const casadi::SX &pi : p) {
      in.push_back(pi);
    }

    // Create coefficient functions
    fc_ = std::make_shared<damotion::casadi::FunctionWrapper<Eigen::VectorXd>>(
        casadi::Function(this->name() + "_A", in, {densify(c)}));
    fb_ = std::make_shared<damotion::casadi::FunctionWrapper<double>>(
        casadi::Function(this->name() + "_b", in, {densify(b)}));

    this->has_grd_ = true;
    // Indicate cost does not have a hessian
    this->has_hes_ = false;
  }
};

}  // namespace optimisation

}  // namespace damotion

#endif /* COSTS_LINEAR_H */
