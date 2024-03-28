#ifndef OSC_OSC_H
#define OSC_OSC_H

#include <Eigen/Core>
#include <algorithm>
#include <casadi/casadi.hpp>
#include <map>
#include <string>

#include "common/profiler.h"
#include "solvers/constraint.h"
#include "solvers/cost.h"
#include "system/constraint.h"
#include "utils/casadi.h"
#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"

namespace opt = damotion::optimisation;
namespace sym = damotion::symbolic;

namespace damotion {
namespace control {
namespace osc {

class EndEffector : public sym::ExpressionVector {
   public:
    EndEffector() { this->resize(3); }
    ~EndEffector() = default;

    void SetName(const std::string &name) { name_ = name; }
    const std::string &name() const { return name_; }

    void SetDimension(const int &dim) { dim_ = dim; }
    const int &Dimension() const { return dim_; }

    /**
     * @brief Constraint
     *
     * @return const ::casadi::SX&
     */
    const ::casadi::SX &Position() const { return (*this)[kPosition]; }
    /**
     * @brief First time derivative of the holonomic constraint
     *
     * @return const ::casadi::SX&
     */
    const ::casadi::SX &Velocity() const { return (*this)[kVelocity]; }

    /**
     * @brief Second time derivative of the holonomic constraint
     *
     * @return const ::casadi::SX&
     */
    const ::casadi::SX &Acceleration() const { return (*this)[kAcceleration]; }

    const Eigen::VectorXd EvalPosition() {
        return Function().getOutput(kPosition);
    }
    const Eigen::VectorXd EvalVelocity() {
        return Function().getOutput(kVelocity);
    }

    void SetPosition(const ::casadi::SX &c) { (*this)[kPosition] = c; }
    void SetVelocity(const ::casadi::SX &dc) { (*this)[kVelocity] = dc; }
    void SetAcceleration(const ::casadi::SX &ddc) {
        (*this)[kAcceleration] = ddc;
    }

   private:
    enum Index { kPosition = 0, kVelocity, kAcceleration };

    int dim_ = 0;
    std::string name_;
};

class HolonomicConstraint : public sym::ExpressionVector {
   public:
    HolonomicConstraint() { this->resize(3); }
    ~HolonomicConstraint() = default;

    void SetName(const std::string &name) { name_ = name; }
    const std::string &name() const { return name_; }

    void SetDimension(const int &dim) { dim_ = dim; }
    const int &Dimension() const { return dim_; }

    /**
     * @brief Constraint
     *
     * @return const ::casadi::SX&
     */
    const ::casadi::SX &Constraint() const { return (*this)[kConstraint]; }
    /**
     * @brief First time derivative of the holonomic constraint
     *
     * @return const ::casadi::SX&
     */
    const ::casadi::SX &ConstraintFirstDerivative() const {
        return (*this)[kFirstDerivative];
    }

    /**
     * @brief Second time derivative of the holonomic constraint
     *
     * @return const ::casadi::SX&
     */
    const ::casadi::SX &ConstraintSecondDerivative() const {
        return (*this)[kSecondDerivative];
    }

    const Eigen::VectorXd EvalConstraint() {
        return Function().getOutput(kConstraint);
    }
    const Eigen::VectorXd EvalConstraintFirstDerivative() {
        return Function().getOutput(kFirstDerivative);
    }

    void SetConstraint(const ::casadi::SX &c) { (*this)[kConstraint] = c; }
    void SetConstraintFirstDerivative(const ::casadi::SX &dc) {
        (*this)[kFirstDerivative] = dc;
    }
    void SetConstraintSecondDerivative(const ::casadi::SX &ddc) {
        (*this)[kSecondDerivative] = ddc;
    }

   private:
    enum Index { kConstraint = 0, kFirstDerivative, kSecondDerivative };

    int dim_ = 0;
    std::string name_;
};

struct TrackingTaskData {
    enum class Type { kTranslational, kRotational, kFull };
    Type type;

    // Tracking gains
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kp;
    // Tracking gains
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kd;

    // Error in pose
    Eigen::VectorXd e;
    Eigen::VectorXd de;

    // Desired pose translational component
    Eigen::Vector3d xr;
    // Desired pose rotational component
    Eigen::Quaterniond qr;

    // Desired pose velocity translational component
    Eigen::Vector3d vr = Eigen::Vector3d::Zero();
    // Desired pose velocity rotational component
    Eigen::Vector3d wr = Eigen::Vector3d::Zero();
};

struct ContactTaskData {
    // Error in pose
    Eigen::VectorXd e;
    Eigen::VectorXd de;

    // Desired pose translational component
    Eigen::Vector3d xr;

    // Whether the point is in contact or not
    bool inContact = false;

    // Contact surface normal
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();

    // Friction coefficient
    double mu = 1.0;
};

Eigen::VectorXd DesiredTrackingTaskAcceleration(TrackingTaskData &data,
                                                EndEffector &ee);

sym::Expression TaskAccelerationErrorCost(const casadi::SX &xacc,
                                          const casadi::SX &xacc_d) {
    sym::Expression obj;
    obj = casadi::SX::dot(xacc - xacc_d, xacc - xacc_d);
    return obj;
}

/**
 * @brief Returns the linear coefficients for the no-slip constraints required
 * of OSC controllers, given they are affine in qacc.
 *
 * @param xacc
 * @param qacc
 * @return sym::ExpressionVector
 */
void NoSlipConstraintCoefficients(const casadi::SX &xacc,
                                  const casadi::SX &qacc, casadi::SX &A,
                                  casadi::SX &b) {
    casadi::SX::linear_coeff(xacc, qacc, A, b, true);
}

optimisation::Constraint LinearisedFrictionConstraint() {
    // Square pyramid approximation
    casadi::SX lambda = casadi::SX::sym("lambda", 3);
    casadi::SX mu = casadi::SX::sym("mu");

    casadi::SX l_x = lambda(0), l_y = lambda(1), l_z = lambda(2);

    // TODO - Add parameter for surface normal

    // Friction cone constraint with square pyramid approximation
    sym::Expression cone;
    cone = casadi::SX(4, 1);
    cone(0) = sqrt(2.0) * l_x + mu * l_z;
    cone(1) = -sqrt(2.0) * l_x - mu * l_z;
    cone(2) = sqrt(2.0) * l_y + mu * l_z;
    cone(3) = -sqrt(2.0) * l_y - mu * l_z;

    cone.SetInputs({lambda}, {mu});
    return optimisation::Constraint(cone, opt::BoundsType::kPositive,
                                    "friction_cone");
}

sym::Expression ComputeConstrainedDynamics(
    sym::Expression &unconstrained_inverse_dynamics,
    std::vector<HolonomicConstraint> &constraints, const casadi::SX &qpos,
    const casadi::SX &qvel) {
    sym::Expression constrained_inverse_dynamics =
        unconstrained_inverse_dynamics;

    // Add effects of holonomic constraints
    for (HolonomicConstraint &c : constraints) {
        // Evaluate Jacobian
        casadi::SX dc = c.ConstraintFirstDerivative();
        casadi::SX J = jacobian(dc, qvel);
        // Create constraint forces
        casadi::SX lam = casadi::SX::sym(c.name() + "_lam", c.Dimension());
        // Add joint-space forces associated with the task
        constrained_inverse_dynamics -= mtimes(J.T(), lam);
    }

    return constrained_inverse_dynamics;
}

// TODO - Place this is a utility
Eigen::Quaternion<double> RPYToQuaterion(const double roll, const double pitch,
                                         const double yaw);

}  // namespace osc
}  // namespace control
}  // namespace damotion

#endif /* OSC_OSC_H */
