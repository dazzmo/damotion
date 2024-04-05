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
    typedef int Id;

    /**
     * @brief Unique ID of the end-effector
     *
     * @return const Id&
     */
    const Id &id() const { return id_; }

    EndEffector(const std::string &name) : name_(name) {
        id_ = CreateId();
        this->resize(3);
    }
    ~EndEffector() = default;

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
    // Dimension of the end-effector
    int dim_ = 0;
    // Name of the end-effector
    std::string name_;

    Id id_;

    Id CreateId() {
        static Id next_id = 0;
        Id id = next_id++;
        return id;
    }
};

/**
 * @brief Create and end-effector data type of a specific class?
 *
 */
class EndEffectorFactory {
   public:
    EndEffectorFactory() = default;
    ~EndEffectorFactory() = default;

    std::shared_ptr<EndEffector> Create(const std::string &name) {
        return std::make_shared<EndEffector>(name);
    }

   private:
};

class HolonomicConstraint : public sym::ExpressionVector {
   public:
    typedef int Id;

    /**
     * @brief Unique ID of the end-effector
     *
     * @return const Id&
     */
    const Id &id() const { return id_; }

    HolonomicConstraint(const std::string &name) : name_(name) {
        id_ = CreateId();
        this->resize(3);
    }
    ~HolonomicConstraint() = default;

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

    Id id_;

    Id CreateId() {
        static Id next_id = 0;
        Id id = next_id++;
        return id;
    }
};

class TaskData {
   public:
    TaskData() = default;
    ~TaskData() = default;
    // Task error
    Eigen::VectorXd e;
    Eigen::VectorXd de;

    // PD Tracking gains
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kp;
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kd;

    virtual Eigen::VectorXd GetPDError() { return Kp * e + Kd * de; }

   private:
};

class TrackingTaskData : public TaskData {
   public:
    enum class Type { kTranslational, kRotational, kFull };

    TrackingTaskData() = default;
    ~TrackingTaskData() = default;

    TrackingTaskData(const std::shared_ptr<EndEffector> &ee, const Type &type)
        : TaskData(), ee_(ee) {
        type_ = type;
        if (type == Type::kTranslational) {
            Kp.resize(3);
            Kd.resize(3);
        } else if (type == Type::kRotational) {
            Kp.resize(3);
            Kd.resize(3);
        } else {
            Kp.resize(6);
            Kd.resize(6);
        }
    }

    EndEffector &GetEndEffector() { return *ee_; }

    // Desired pose translational component
    Eigen::Vector3d xr;
    // Desired pose rotational component
    Eigen::Quaterniond qr;

    // Desired pose velocity translational component
    Eigen::Vector3d vr = Eigen::Vector3d::Zero();
    // Desired pose velocity rotational component
    Eigen::Vector3d wr = Eigen::Vector3d::Zero();

    /**
     * @brief Compute the desired tracking task acceleration as a PD error
     * metric on the task position and velocity errors
     *
     * @return Eigen::VectorXd
     */
    void DesiredTrackingTaskAcceleration();

    Eigen::VectorXd GetPDError() override {
        DesiredTrackingTaskAcceleration();
        return Kp * e + Kd * de;
    }

   private:
    Type type_;
    std::shared_ptr<EndEffector> ee_;
};

class TrackingTaskDataFactory {
   public:
    TrackingTaskDataFactory() = default;
    ~TrackingTaskDataFactory() = default;

    std::unique_ptr<TrackingTaskData> Create(
        std::shared_ptr<EndEffector> &ee, const TrackingTaskData::Type &type) {
        return std::make_unique<TrackingTaskData>(ee, type);
    }

   private:
};

class ContactTaskData : public TaskData {
   public:
    ContactTaskData() = default;
    ~ContactTaskData() = default;

    ContactTaskData(const std::shared_ptr<EndEffector> &ee)
        : TaskData(), ee_(ee) {}

    // Desired pose translational component
    Eigen::Vector3d xr;

    // Whether the point is in contact or not
    bool inContact = false;

    // Contact surface normal
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();

    // Friction coefficient
    double mu = 1.0;

   private:
    std::shared_ptr<EndEffector> ee_;
};

class ContactTaskDataFactory {
   public:
    ContactTaskDataFactory() = default;
    ~ContactTaskDataFactory() = default;

    std::unique_ptr<ContactTaskData> Create(std::shared_ptr<EndEffector> &ee) {
        return std::make_unique<ContactTaskData>(ee);
    }

   private:
};

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
    return optimisation::Constraint("friction_cone", cone,
                                    opt::BoundsType::kPositive);
}

// TODO - Place this is a utility
Eigen::Quaternion<double> RPYToQuaterion(const double roll, const double pitch,
                                         const double yaw);

}  // namespace osc
}  // namespace control
}  // namespace damotion

#endif /* OSC_OSC_H */
