#ifndef CONSTRAINTS_BOUNDING_BOX_H
#define CONSTRAINTS_BOUNDING_BOX_H

#include "damotion/optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

class BoundingBoxConstraint : public Constraint {
 public:
  BoundingBoxConstraint(const std::string &name, const int &nx,
                        const Eigen::VectorXd &lb, const Eigen::VectorXd &ub)
      : Constraint(nx, 2 * nx) {
    assert(lb.rows() == ub.rows() && "lb and ub must be same dimension!");
    // Create output vector and derivative
    f_ = GenericEigenMatrix(2 * nx, 1);
    Eigen::SparseMatrix<double> D(2 * nx, nx);
    // Create diagonal matrix with +1 on the first half of the diagonal and -1
    // on the second half Update bounds
    for (int i = 0; i < nx; ++i) {
      D.coeffRef(i, i) = 1.0;
      D.coeffRef(nx + i, i) = -1.0;
    }
    D.makeCompressed();
    d_ = GenericEigenMatrix(D);

    this->SetBounds(lb, ub);
  }

  void EvalImpl() override {
    // Get vector
    Eigen::Ref<const Eigen::VectorXd> x =
        Eigen::Map<const Eigen::VectorXd>(in_[0], this->ny());
    // Compute bounds
    f_.toVectorXdRef() << x - LowerBound(), UpperBound() - x;
  }

  void DerivativeImpl() override {
    // Constant matrix, no need to update
  }

  const GenericEigenMatrix &GetOutput(const int &i) const override {
    assert(i == 0 && "Bounding box constraints only have one output!");
    return f_;
  }

  const GenericEigenMatrix &GetDerivative(const int &i) const override {
    assert(i == 0 && "Bounding box constraints only have one output!");
    return d_;
  }

 private:
  GenericEigenMatrix f_;
  GenericEigenMatrix d_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_BOUNDING_BOX_H */
