#ifndef SYSTEM_CONSTRAINT_H
#define SYSTEM_CONSTRAINT_H

#include <casadi/casadi.hpp>

namespace system {

class Constraint {
   public:
    /**
     * @brief Empty constructor
     *
     */
    Constraint() : nc_(0) {}
    /**
     * @brief Construct a new Constraint object
     *
     * @param nc Dimension of the constraint
     */
    Constraint(int nc) : nc_(nc) {}

    ~Constraint() = default;

    casadi::Function vec() { return f_; }
    casadi::Function jac() { return df_; }

   protected:
    void setVec(casadi::Function &f) { f_ = f; }
    void setJac(casadi::Function &f) { df_ = f; }

   private:
    int nc_;

    // Constraint evaluation function
    casadi::Function f_;
    // Constraint jacobian function
    casadi::Function df_;
};

/**
 * @brief Configuration-dependent constraint of the form \f$ h = c(q) \f$
 *
 */
class HolonomicConstraint : public Constraint {
   public:
    /**
     * @brief Empty constructor
     *
     */
    HolonomicConstraint() : Constraint() {}

    /**
     * @brief Construct a new Holonomic Constraint object
     *
     * @param nc Dimension of the constraint
     * @param nq Dimension of the configuration
     * @param nv Dimension of the tangent space
     */
    HolonomicConstraint(int nc, int nq, int nv)
        : Constraint(nc), nq_(nq), nv_(nv) {}

    ~HolonomicConstraint() = default;

    /**
     * @brief The second time derivative of the constraint \f$ c(q) = 0 \f$ such
     * that \f$ \ddot{c(q)} = \dot{J}(q) \dot{q} + J(q) \ddot{q} \f$.
     *
     * @return casadi::Function
     */
    casadi::Function secondTimeDerivative() { return ddf_; }

   protected:
    void setSecondTimeDerivative(casadi::Function &f) { ddf_ = f; }

   private:
    // Dimension of configuration space
    int nq_;
    // Dimension of the tangent space
    int nv_;

    casadi::Function ddf_;
};

}  // namespace system
#endif /* SYSTEM_CONSTRAINT_H */
