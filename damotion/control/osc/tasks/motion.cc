#include "damotion/control/osc/tasks/motion.h"

namespace damotion {
namespace control {
namespace osc {

void PositionTask::ComputeMotionError() {
  damotion::common::Profiler profiler("PositionTask::ComputeMotionError");

  e_ = pos() - GetReference().x;
  de_ = vel() - GetReference().v;
}

void OrientationTask::ComputeMotionError() {
  damotion::common::Profiler profiler("OrientationTask::ComputeMotionError");

  // Get rotational component
  Eigen::Quaterniond q;
  // TODO - Find a way to conventionalise the quaternions, Eigen also seems to
  // do the [x, y, z, w] format
  q.w() = pos()[0];
  q.x() = pos()[1];
  q.y() = pos()[2];
  q.z() = pos()[3];
  // Compute rotational error
  Eigen::Matrix3d R =
      GetReference().q.normalized().toRotationMatrix().transpose() *
      q.normalized().toRotationMatrix();
  double theta = 0.0;
  Eigen::Vector3d elog;
  elog = pinocchio::log3(R, theta);

  // Compute rate of error
  Eigen::Matrix3d Jlog;
  pinocchio::Jlog3(theta, elog, Jlog);
  e_ = elog;
  de_ = Jlog * vel() - GetReference().w;
}

void Pose6DTask::ComputeMotionError() {
  damotion::common::Profiler profiler("Pose6DTask::ComputeMotionError");

  // Convert to 6D pose
  Eigen::Vector3d x = pos().topRows(3);
  Eigen::Vector4d qv = pos().bottomRows(4);
  Eigen::Quaterniond q(qv);
  pinocchio::SE3 se3r(GetReference().q.toRotationMatrix(), GetReference().x),
      se3(q.toRotationMatrix(), x);

  e_ = pinocchio::log6(se3r.actInv(se3)).toVector();
  Eigen::Matrix<double, 6, 6> Jlog;
  pinocchio::Jlog6(se3r.actInv(se3), Jlog);
  de_ = Jlog * vel();
  de_.topRows(3) -= GetReference().v;     // Translational component
  de_.bottomRows(3) -= GetReference().w;  // Rotational component
}

}  // namespace osc
}  // namespace control
}  // namespace damotion
