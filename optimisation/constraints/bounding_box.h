#ifndef CONSTRAINTS_BOUNDING_BOX_H
#define CONSTRAINTS_BOUNDING_BOX_H

#include "optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

template <typename MatrixType>
class BoundingBoxConstraint : public ConstraintBase<MatrixType> {
   public:
    BoundingBoxConstraint(const std::string &name,
                              const Eigen::VectorXd &lb,
                              const Eigen::VectorXd &ub)
        : ConstraintBase<MatrixType>(name, "bounding_box_constraint") {
        assert(lb.rows() == ub.rows() && "lb and ub must be same dimension!");

        int n = lb.rows();
        // Resize the constraint
        this->Resize(n, n, 0);
        // Update bounds
        this->UpdateBounds(lb, ub);

        casadi::SX x = casadi::SX::sym("x", n);

        casadi::Function f = casadi::Function(this->name(), {x}, {densify(x)});
        casadi::Function fjac = casadi::Function(this->name() + "_jac", {x},
                                                 {densify(jacobian(x, x))});

        this->SetConstraintFunction(
            std::make_shared<utils::casadi::FunctionWrapper<Eigen::VectorXd>>(
                f));
        this->SetJacobianFunction(
            std::make_shared<utils::casadi::FunctionWrapper<MatrixType>>(fjac));
    }

   private:
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_BOUNDING_BOX_H */
