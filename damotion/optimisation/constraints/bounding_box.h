#ifndef CONSTRAINTS_BOUNDING_BOX_H
#define CONSTRAINTS_BOUNDING_BOX_H

#include "damotion/optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

class BoundingBoxConstraint : public ConstraintBase {
 public:
  BoundingBoxConstraint(const std::string &name, const Eigen::VectorXd &lb,
                        const Eigen::VectorXd &ub) {
    assert(lb.rows() == ub.rows() && "lb and ub must be same dimension!");
    // Create constraint
    casadi::SX c;

    int n = lb.rows();

    // Resize the constraint
    this->Resize(n, n, 0);
    // Update bounds
    this->SetBounds(lb, ub);
  }

 private:
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_BOUNDING_BOX_H */
