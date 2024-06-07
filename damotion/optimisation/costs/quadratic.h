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
class QuadraticCost : public CostBase {
 public:
  using UniquePtr = std::unique_ptr<QuadraticCost>;
  using SharedPtr = std::shared_ptr<QuadraticCost>;

  QuadraticCost(const std::string &name, const Eigen::MatrixXd &A,
                const Eigen::VectorXd &b, const double &c, bool jac = true,
                bool hes = true) {
    // Cost
    casadi::DM Ad, bd;
    casadi::SX csx = c;
    damotion::casadi::toCasadi(A, Ad);
    damotion::casadi::toCasadi(b, bd);
    casadi::SX Asx = Ad, bsx = bd;
    ConstructConstraint(name, Asx, bsx, csx, {}, jac, hes);
  }

  QuadraticCost(const std::string &name, const casadi::SX &A,
                const casadi::SX &b, const casadi::SX &c,
                const casadi::SXVector &p, bool jac = true, bool hes = true) {
    ConstructConstraint(name, A, b, c, p, jac, hes);
  }

  QuadraticCost(const std::string &name, const casadi::SX &ex,
                const casadi::SXVector &x, const casadi::SXVector &p,
                bool jac = true, bool hes = true) {
    casadi::SXVector in = {};
    // Extract quadratic form
    casadi::SX A, b, c;
    casadi::SX::quadratic_coeff(ex, x[0], A, b, c, true);

    // Remove factor of two from hessian
    A *= 0.5;

    ConstructConstraint(name, A, b, c, p, jac, hes);
  }

  /**
   * @brief Lower triangle representation of the quadratic cost Hessian
   *
   * @return const MatrixType&
   */
  const GenericEigenMatrix &A() const { return fA_->getOutput(0); }

  const GenericEigenMatrix &b() const { return fb_->getOutput(0); }

  const GenericEigenMatrix &c() const { return fc_->getOutput(0); }

 private:
  common::Function::SharedPtr fA_;
  common::Function::SharedPtr fb_;
  common::Function::SharedPtr fc_;

  void ConstructConstraint(const std::string &name, const casadi::SX &A,
                           const casadi::SX &b, const casadi::SX &c,
                           const casadi::SXVector &p, bool jac = true,
                           bool hes = true, bool sparse = false) {
    this->SetName(name);

    casadi::SXVector in = {};
    // Linear cost
    casadi::SX x = casadi::SX::sym("x", A.rows());
    casadi::SX cost = mtimes(x.T(), mtimes(A, x)) + mtimes(b.T(), x) + c;
    in.push_back(x);
    for (const casadi::SX &pi : p) {
      in.push_back(pi);
    }

    // Create coefficient functions
    casadi::SX Atmp = casadi::SX::tril(A), btmp = b, ctmp = c;
    if (!sparse) {
      Atmp = densify(Atmp);
      btmp = densify(btmp);
      ctmp = densify(ctmp);
    }
    // Create functions
    fA_ = std::make_shared<damotion::casadi::FunctionWrapper>(
        casadi::Function(this->name() + "_A", in, {Atmp}));
    fb_ = std::make_shared<damotion::casadi::FunctionWrapper>(
        casadi::Function(this->name() + "_b", in, {btmp}));
    fc_ = std::make_shared<damotion::casadi::FunctionWrapper>(
        casadi::Function(this->name() + "_c", in, {ctmp}));

    // TODO

    this->has_grd_ = true;
    this->has_hes_ = true;
  }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* COSTS_QUADRATIC_H */
