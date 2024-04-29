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
        this->SetBounds(lb, ub);       
    }

    /**
     * @brief Evaluate the constraint and Jacobian (optional) given input
     * variables x and parameters p.
     *
     * @param x
     * @param p
     * @param jac Flag for computing the Jacobian
     */
    void eval(const common::InputRefVector &x, const common::InputRefVector &p,
              bool jac = true) const override {
        // Evaluate the constraint
        c_ << x[0] - this->LowerBound(), x[0] + this->UpperBound();
    }

    /**
     * @brief Returns the most recent evaluation of the constraint
     *
     * @return const Eigen::VectorXd&
     */
    const Eigen::VectorXd &Vector() const override {
        VLOG(10) << this->name() << " Vector = " << c_;
        return c_;
    }
    /**
     * @brief The Jacobian of the constraint with respect to the i-th variable
     * vector
     *
     * @param i
     * @return const MatrixType&
     */
    const MatrixType &Jacobian(const int &i) const override {
        assert(i != 0 && "Bounding box constraint only has one Jacobian!");
        VLOG(10) << this->name() << " Jacobian " << i << " = " << J_;
        return J_;
    }

   private:
    // Constraint Vector
    mutable Eigen::VectorXd c_;
    // Jacobian
    mutable MatrixType J_;
    
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_BOUNDING_BOX_H */
