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

Eigen::VectorXd DesiredTrackingTaskAcceleration(TrackingTaskData &data,
                                                EndEffector &ee) {
    damotion::common::Profiler profiler(
        "TrackingTask::ComputeDesiredAcceleration");

    const Eigen::VectorXd &xpos = ee.EvalPosition(), &xvel = ee.EvalVelocity();

    if (data.type == TrackingTaskData::Type::kTranslational) {
        data.e = xpos - data.xr;
        data.de = xvel - data.vr;
    } else if (data.type == TrackingTaskData::Type::kRotational) {
        // Get rotational component
        Eigen::Vector4d qv = xpos.bottomRows(4);
        Eigen::Quaterniond q(qv);
        // Compute rotational error
        Eigen::Matrix3d R =
            data.qr.toRotationMatrix().transpose() * q.toRotationMatrix();
        double theta = 0.0;
        Eigen::Vector3d elog;
        elog = pinocchio::log3(R, theta);
        // Compute rate of error
        Eigen::Matrix3d Jlog;
        pinocchio::Jlog3(theta, elog, Jlog);
        data.e = elog;
        data.de = Jlog * xvel - data.wr;
    } else {
        // Convert to 6D pose
        Eigen::Vector4d qv = xpos.bottomRows(4);
        Eigen::Quaterniond q(qv);
        Eigen::Vector3d x = xpos.topRows(3);
        pinocchio::SE3 se3r(data.qr.toRotationMatrix(), data.xr),
            se3(q.toRotationMatrix(), x);

        data.e = pinocchio::log6(se3r.actInv(se3)).toVector();
        Eigen::Matrix<double, 6, 6> Jlog;
        pinocchio::Jlog6(se3r.actInv(se3), Jlog);
        data.de = Jlog * xvel;
        data.de.topRows(3) -= data.vr;     // Translational component
        data.de.bottomRows(3) -= data.wr;  // Rotational component
    }

    // Return desired acceleration as a PD cost on task error
    return data.Kp * data.e + data.Kd * data.de;
}

}  // namespace osc
}  // namespace control
}  // namespace damotion