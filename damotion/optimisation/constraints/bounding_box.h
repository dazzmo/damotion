#ifndef CONSTRAINTS_BOUNDING_BOX_H
#define CONSTRAINTS_BOUNDING_BOX_H

#include "damotion/optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

/**
 * @brief Represents the function for the constraint lb < f(x, p) < ub
 *
 */
class BoundingBoxFunction : public common::Function {
  // Override the function and include derivative data
 public:
  BoundingBoxFunction(const size_t &n) {
    // Create constraint vector
    out_.push_back(GenericEigenMatrix(n, 1));
    // Create jacobian matrix
    Eigen::SparseMatrix<double> I(n, n);
    I.setIdentity();
    out_.push_back(GenericEigenMatrix(I));
  }

  void evalImpl(const std::vector<ConstVectorRef> &input, bool check = false) {
    // Adjust constraint vector
    out_[0].toVectorXdRef() << input[0];
    // Jacobian is constant
  }

  const GenericEigenMatrix &GetOutputImpl(const size_t &i) { return out_[i]; }

 private:
  std::vector<GenericEigenMatrix> out_;
};

class BoundingBoxConstraint : public Constraint {
 public:
  BoundingBoxConstraint(const std::string &name, const int &nx,
                        const Eigen::VectorXd &lb, const Eigen::VectorXd &ub)
      : Constraint("bb", std::make_shared<BoundingBoxFunction>(),
                   BoundsType::kCustom) {
    assert(lb.rows() == ub.rows() && "lb and ub must be same dimension!");
    // Create output vector and derivative
    this->setBounds(lb, ub);
  }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_BOUNDING_BOX_H */
