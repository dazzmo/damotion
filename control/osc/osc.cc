#include "control/osc/osc.h"

namespace damotion {
namespace control {
namespace osc {

Eigen::Quaterniond RPYToQuaterion(const double roll, const double pitch,
                                  const double yaw) {
    Eigen::AngleAxisd r = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()),
                      p = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()),
                      y = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    return Eigen::Quaterniond(r * p * y);
}

void TrackingTaskData::DesiredTrackingTaskAcceleration() {
    damotion::common::Profiler profiler(
        "TrackingTask::ComputeDesiredAcceleration");

    const Eigen::VectorXd &xpos = ee_->EvalPosition(),
                          &xvel = ee_->EvalVelocity();

    if (type_ == Type::kTranslational) {
        e = xpos.topRows(3) - xr;
        de = xvel.topRows(3) - vr;
    } else if (type_ == Type::kRotational) {
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
}

}  // namespace osc
}  // namespace control
}  // namespace damotion