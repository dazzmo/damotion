#include "common/trajectory/trajectory.h"

namespace damotion {
namespace trajectory {

bool IsBoundedWithinBoxLimits(Trajectory<double> &trajectory,
                              const Eigen::VectorXd &xl,
                              const Eigen::VectorXd &xu, double resolution) {
  double dt = trajectory.duration() / (resolution - 1);
  for (double t = trajectory.t0(); t <= trajectory.tf(); t += dt) {
    Eigen::VectorXd x = trajectory.eval(t);
    if ((xl + x).minCoeff() < 0 || (xu - x).minCoeff() < 0) {
      return false;
    }
  }
  return true;
}

template <>
const Eigen::VectorXd &DiscreteTrajectory<double>::eval(
    double &t, const double &precision) const {
  // Move through entries in time
  int idx = 0;
  for (int i = 0; i < t_.size(); ++i) {
    if (t < t_[idx] - precision) break;
    idx++;
  }

  return x_[idx];
}

}  // namespace trajectory
}  // namespace damotion
