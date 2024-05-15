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
template <typename MatrixType>
class QuadraticCost : public CostBase<MatrixType> {
 public:
  using UniquePtr = std::unique_ptr<QuadraticCost<MatrixType>>;
  using SharedPtr = std::shared_ptr<QuadraticCost<MatrixType>>;

  QuadraticCost(const std::string &name, const Eigen::MatrixXd &A,
                const Eigen::VectorXd &b, const double &c, bool jac = true,
                bool hes = true)
      : CostBase<MatrixType>(name, "quadratic_cost") {
    // Cost
    casadi::DM Ad, bd;
    casadi::SX csx = c;
    damotion::casadi::toCasadi(A, Ad);
    damotion::casadi::toCasadi(b, bd);
    casadi::SX Asx = Ad, bsx = bd;
    ConstructConstraint(Asx, bsx, csx, {}, jac, hes);
  }

  QuadraticCost(const std::string &name, const casadi::SX &A,
                const casadi::SX &b, const casadi::SX &c,
                const casadi::SXVector &p, bool jac = true, bool hes = true)
      : CostBase<MatrixType>(name, "quadratic_cost") {
    ConstructConstraint(A, b, c, p, jac, hes);
  }

  QuadraticCost(const std::string &name, const sym::Expression &ex,
                bool jac = true, bool hes = true)
      : CostBase<MatrixType>(name, "quadratic_cost") {
    casadi::SXVector in = {};
    // Extract quadratic form
    casadi::SX A, b, c;
    casadi::SX::quadratic_coeff(ex, ex.Variables()[0], A, b, c, true);

    // Remove factor of two from hessian
    A *= 0.5;

    ConstructConstraint(A, b, c, ex.Parameters(), jac, hes);
  }

  /**
   * @brief Lower triangle representation of the quadratic cost Hessian
   *
   * @return const MatrixType&
   */
  const MatrixType &A() const { return fA_->getOutput(0); }

  const Eigen::VectorXd &b() const { return fb_->getOutput(0); }

  const double &c() const { return fc_->getOutput(0); }

  /**
   * @brief Evaluate the constraint and Jacobian (optional) given input
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
    fA_->call(p);
    fb_->call(p);
    fc_->call(p);

    // Evaluate the constraint
    this->obj_ = x[0].dot(A().template selfadjointView<Eigen::Lower>() * x[0]) +
                 b().dot(x[0]) + c();
    if (grd) {
      this->grd_ =
          2.0 * static_cast<Eigen::VectorXd>(
                    A().template selfadjointView<Eigen::Lower>() * x[0]) +
          b();
    }
  }

  void eval_hessian(const common::InputRefVector &x,
                    const common::InputRefVector &p) const override {
    VLOG(10) << this->name() << " eval_hessian()";
    this->hes_ = 2.0 * A();
  }

 private:
  std::shared_ptr<common::Function<MatrixType>> fA_;
  std::shared_ptr<common::Function<Eigen::VectorXd>> fb_;
  std::shared_ptr<common::Function<double>> fc_;

  void ConstructConstraint(const casadi::SX &A, const casadi::SX &b,
                           const casadi::SX &c, const casadi::SXVector &p,
                           bool jac = true, bool hes = true) {
    casadi::SXVector in = {};
    // Linear cost
    casadi::SX x = casadi::SX::sym("x", A.rows());
    casadi::SX cost = mtimes(x.T(), mtimes(A, x)) + mtimes(b.T(), x) + c;
    in.push_back(x);
    for (const casadi::SX &pi : p) {
      in.push_back(pi);
    }

    // Create coefficient functions
    casadi::SX A_lt = casadi::SX::tril(A);
    if (std::is_same<MatrixType, Eigen::SparseMatrix<double>>::value) {
      fA_ = std::make_shared<damotion::casadi::FunctionWrapper<MatrixType>>(
          casadi::Function(this->name() + "_A", in, {A_lt}));
    } else {
      fA_ = std::make_shared<damotion::casadi::FunctionWrapper<MatrixType>>(
          casadi::Function(this->name() + "_A", in, {densify(A_lt)}));
    }
    fb_ = std::make_shared<damotion::casadi::FunctionWrapper<Eigen::VectorXd>>(
        casadi::Function(this->name() + "_b", in, {densify(b)}));
    fc_ = std::make_shared<damotion::casadi::FunctionWrapper<double>>(
        casadi::Function(this->name() + "_c", in, {densify(c)}));

    // Set hessian structure
    common::Sparsity sparsity(A_lt.sparsity());

    this->grd_ = Eigen::VectorXd::Zero(A.rows());
    this->hes_ = common::CreateSparseEigenMatrix(sparsity);

    this->has_grd_ = true;
    this->has_hes_ = true;
  }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* COSTS_QUADRATIC_H */
