#ifndef SYSTEM_CONSTRAINT_H
#define SYSTEM_CONSTRAINT_H

#include <casadi/casadi.hpp>

#include "system/controlled.h"
#include "utils/pinocchio_model.h"

namespace damotion {
namespace system {

class Constraint {
   public:
    /**
     * @brief Empty constructor
     *
     */
    Constraint() : name_(""), nc_(0) {}
    /**
     * @brief Construct a new Constraint object
     *
     * @param name Name of the constraint
     * @param nc Dimension of the constraint
     */
    Constraint(const std::string &name, int nc) : name_(name), nc_(nc) {}

    Constraint(const std::string &name, const casadi::Function &constraint,
               const casadi::Function &jacobian);

    ~Constraint() = default;

    /**
     * @brief Name of the constraint
     *
     * @return const std::string&
     */
    const std::string &name() const { return name_; }

    /**
     * @brief Dimension of the constraint
     *
     * @return const int
     */
    const int nc() const { return nc_; }

    /**
     * @brief Function that computs the constraint
     *
     * @return casadi::Function
     */
    casadi::Function constraint() { return f_; }

    /**
     * @brief Function for the Jacobian of the constraint
     *
     * @return casadi::Function
     */
    casadi::Function jacobian() { return df_; }

    void setConstraint(const casadi::SX &c, const casadi::SXVector &in,
    const casadi::StringVector &name_in = {},
    const casadi::StringVector &name_out = {});
    void setJacobian(const casadi::SX &J, const casadi::SXVector &in);

   protected:
    void setConstraint(const casadi::Function &f) { f_ = f; }
    void setJacobian(const casadi::Function &f) { df_ = f; }

   private:
    // Dimension of the constraint
    int nc_;

    // Name of the constraint
    std::string name_;

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
     * @param name Name of the constraint
     * @param nc Dimension of the constraint
     * @param nq Dimension of the configuration
     * @param nv Dimension of the tangent space
     */
    HolonomicConstraint(const std::string &name, int nc, int nq, int nv)
        : Constraint(name, nc), nq_(nq), nv_(nv) {}

    /**
     * @brief Construct a new holonomic constraint by providing the functions
     * that evaluate the constraint, jacobian and first and second time
     * derivatives.
     *
     * @param name
     * @param constraint
     * @param jacobian
     * @param first_time_derivative
     * @param second_time_derivative
     */
    HolonomicConstraint(const std::string &name,
                        const casadi::Function &constraint,
                        const casadi::Function &jacobian,
                        const casadi::Function &first_time_derivative,
                        const casadi::Function &second_time_derivative);

    /**
     * @brief Construct a new Holonomic Constraint object based on the
     * expressions created for the constraint, derivatives and Jacobians
     *
     * @param name
     * @param c
     * @param dc
     * @param ddc
     * @param J
     * @param qpos
     * @param qvel
     * @param qacc
     */
    HolonomicConstraint(const std::string &name, const casadi::SX &c,
                        const casadi::SX &dc, const casadi::SX &ddc,
                        const casadi::SX &J, const casadi::SX &qpos,
                        const casadi::SX &qvel, const casadi::SX &qacc,
                        const casadi::SXVector &par = {});

    ~HolonomicConstraint() = default;

    /**
     * @brief Dimension of the configuration vector
     *
     * @return const int&
     */
    const int &nq() const { return nq_; }

    /**
     * @brief Dimension of the tangent-space vector
     *
     * @return const int&
     */
    const int &nv() const { return nv_; }

    /**
     * @brief The first time derivative of the constraint \f$ c(q) = 0 \f$ such
     * that \f$ \dot{c(q)} = J(q) \dot{q} \f$.
     *
     * @return casadi::Function
     */
    casadi::Function firstTimeDerivative() { return df_; }

    /**
     * @brief The second time derivative of the constraint \f$ c(q) = 0 \f$ such
     * that \f$ \ddot{c(q)} = \dot{J}(q) \dot{q} + J(q) \ddot{q} \f$.
     *
     * @return casadi::Function
     */
    casadi::Function secondTimeDerivative() { return ddf_; }

   protected:
    void setFirstTimeDerivative(const casadi::Function &f) { df_ = f; }
    void setSecondTimeDerivative(const casadi::Function &f) { ddf_ = f; }

   private:
    // Dimension of configuration space
    int nq_;
    // Dimension of the tangent space
    int nv_;

    casadi::Function df_;
    casadi::Function ddf_;
};

/**
 * @brief Computes the forward dynamics of the system subject to the holonomic
 * constraints provided in constraints. Outputs a new dynamics function with new
 * constraint forces f that constrain the forward dynamics to the constraints
 * imposed by constraints.
 *
 * @param system
 * @param constraints
 * @return casadi::Function
 */
casadi::Function constrainedDynamics(
    SecondOrderControlledSystem &system,
    std::vector<HolonomicConstraint> &constraints);

/**
 * @brief Computes the inverse dynamics of the system subject to the holonomic
 * constraints provided in constraints. Outputs a new dynamics function with new
 * constraint forces f that constrain the inverse dynamics to the constraints
 * imposed by constraints.
 *
 * @param system
 * @param constraints
 * @return casadi::Function
 */
casadi::Function constrainedInverseDynamics(
    SecondOrderControlledSystem &system,
    std::vector<HolonomicConstraint> &constraints);

}  // namespace system
}  // namespace damotion

#endif /* SYSTEM_CONSTRAINT_H */
