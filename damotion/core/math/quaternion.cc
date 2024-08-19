#include "damotion/core/math/quaternion.h"

namespace damotion {
namespace math {

Eigen::Quaternion<double> RPYToQuaterion(const double roll, const double pitch,
                                         const double yaw) {
  Eigen::AngleAxis<double> r = Eigen::AngleAxis<double>(
                               roll, Eigen::Vector3d::UnitX()),
                           p = Eigen::AngleAxis<double>(
                               pitch, Eigen::Vector3d::UnitY()),
                           y = Eigen::AngleAxis<double>(
                               yaw, Eigen::Vector3d::UnitZ());
  return Eigen::Quaternion<double>(r * p * y);
}

}  // namespace math
}  // namespace damotion
