#ifndef COSTS_QUADRATIC_H
#define COSTS_QUADRATIC_H
#include <Eigen/Core>

#include "damotion/optimisation/costs/base.h"

namespace damotion {
namespace optimisation {
/**
 * @brief A cost of the form 0.5 x^T A x + b^T x + c
 *
 */
class QuadraticCost : public Cost {
 public:
  using UniquePtr = std::unique_ptr<QuadraticCost>;
  using SharedPtr = std::shared_ptr<QuadraticCost>;

  QuadraticCost(const std::string &name, const casadi::SX &A,
                const casadi::SX &b, const casadi::SX &c, const casadi::SX &x,
                const casadi::SXVector &p)
      : Cost(mtimes(mtimes(x.T(), A), x) + mtimes(b.T(), x) + c, {x}, p,
             false) {}

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
  common::Function::SharedPtr fA_;
  common::Function::SharedPtr fb_;
  common::Function::SharedPtr fc_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* COSTS_QUADRATIC_H */
