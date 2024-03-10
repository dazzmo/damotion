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

    casadi::Function vec() { return vecImpl(); }
    casadi::Function jac() { return jacImpl(); }

   protected:
    virtual casadi::Function vecImpl() = 0;
    virtual casadi::Function jacImpl() = 0;

   private:
    int nc_;
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
    casadi::Function secondTimeDerivative() {
        return secondTimeDerivativeImpl();
    }

   protected:
    virtual casadi::Function vecImpl() = 0;
    virtual casadi::Function jacImpl() = 0;
    virtual casadi::Function secondTimeDerivativeImpl() = 0;

   private:
    // Dimension of configuration space
    int nq_;
    // Dimension of the tangent space
    int nv_;
};

}  // namespace system
#endif /* SYSTEM_CONSTRAINT_H */
