#ifndef CONSTRAINTS_BOUNDING_BOX_H
#define CONSTRAINTS_BOUNDING_BOX_H

#include "damotion/optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

class BoundingBoxConstraint : public Constraint {
 public:
  BoundingBoxConstraint(const std::string &name, const int &nx,
                        const ::casadi::SX &lb, const ::casadi::SX &ub,
                        const ::casadi::SX &x, const ::casadi::SX &p)
      : Constraint("bb", std::make_shared<BoundingBoxFunction>(),
                   Bounds::Type::kCustom) {
    assert(lb.rows() == ub.rows() && "lb and ub must be same dimension!");
    // Create output vector and derivative
    this->setBounds(lb, ub);
  }

  void eval(const common::Function::InputVector &x,
            const common::Function::InputVector &p, bool jac) {
    // Evaluate the constraints based on the
    common::Function::InputVector in = {};
    for (const auto &xi : x) in.push_back(xi);
    for (const auto &pi : p) in.push_back(pi);
    // Perform evaluation depending on what method is used
    // TODO - Evaluate the constraint within the provided bounds
  }

 private:
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_BOUNDING_BOX_H */
