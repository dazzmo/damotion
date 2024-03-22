#include "control/osc/osc.h"

namespace damotion {
namespace control {

Eigen::Quaterniond RPYToQuaterion(const double roll, const double pitch,
                                  const double yaw) {
    Eigen::AngleAxisd r = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()),
                      p = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()),
                      y = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    return Eigen::Quaterniond(r * p * y);
}

void OSCController::UpdateProgramParameters() {
    damotion::common::Profiler profiler(
        "OSCController::UpdateProgramParameters");

    // Adjust bounds on contact forces depending on contact states
    for (auto &p : contact_tasks_) {
        ContactTask &task = p.second;
        // Get index of contact forces in optimisation variable
        int idx = GetVariableIndex("lam") + task.ConstraintForceIndex();
        // Update conditions on end-effectors
        if (task.inContact) {
            // Update bounds for lambda
            DecisionVariablesUpperBound().middleRows(idx, task.Dimension())
                << 1e8,
                1e8, 1e8;
            DecisionVariablesLowerBound().middleRows(idx, task.Dimension())
                << -1e8,
                -1e8, 0.0;
        } else {
            // No contact forces
            DecisionVariablesUpperBound()
                .middleRows(idx, task.Dimension())
                .setZero();
            DecisionVariablesLowerBound()
                .middleRows(idx, task.Dimension())
                .setZero();
        }
    }

    // Update tracking costs
    for (auto &p : tracking_tasks_) {
        TrackingTask &task = p.second;
        SetParameter(p.first + "_xacc_d", task.ComputeDesiredAcceleration());
    }
}

Eigen::VectorXd OSCController::TrackingTask::ComputeDesiredAcceleration() {
    damotion::common::Profiler profiler(
        "OSCController::TrackingTask::ComputeDesiredAcceleration");
    // Evaluate the task with current parameters for the functions
    Function().call();
    // Get task position and velocity
    Eigen::VectorXd xpos = Function().getOutput(0);
    Eigen::VectorXd xvel = Function().getOutput(1);

    std::cout << xpos.transpose() << std::endl;
    std::cout << xvel.transpose() << std::endl;

    if (type == Type::kTranslational) {
        e = xpos - xr;
        de = xvel - vr;
    } else if (type == Type::kRotational) {
        // Get rotational component
        Eigen::Vector4d qv = xpos.bottomRows(4);
        Eigen::Quaterniond q(qv);
        // Compute rotational error
        Eigen::Matrix3d R =
            qr.toRotationMatrix().transpose() * q.toRotationMatrix();
        double theta = 0.0;
        Eigen::Vector3d elog;
        elog = pinocchio::log3(R, theta);
        // Compute rate of error
        Eigen::Matrix3d Jlog;
        pinocchio::Jlog3(theta, elog, Jlog);
        e = elog;
        de = Jlog * xvel - wr;
    } else {
        // Convert to 6D pose
        Eigen::Vector4d qv = xpos.bottomRows(4);
        Eigen::Quaterniond q(qv);
        Eigen::Vector3d x = xpos.topRows(3);
        pinocchio::SE3 se3r(qr.toRotationMatrix(), xr),
            se3(q.toRotationMatrix(), x);

        e = pinocchio::log6(se3r.actInv(se3)).toVector();
        Eigen::Matrix<double, 6, 6> Jlog;
        pinocchio::Jlog6(se3r.actInv(se3), Jlog);
        de = Jlog * xvel;
        de.topRows(3) -= vr;     // Translational component
        de.bottomRows(3) -= wr;  // Rotational component
    }

    // Return desired acceleration as a PD cost on task error
    return Kp * e + Kd * de;
}

}  // namespace control
}  // namespace damotion