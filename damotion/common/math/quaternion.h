#ifndef MATH_QUATERNION_H
#define MATH_QUATERNION_H

#include <Eigen/Core>
#include <Eigen/Dense>

namespace damotion {
namespace math {

/**
 * @brief Converts Euler angles of roll, pitch and yaw in RPY order to a
 * quaternion.
 *
 * @param roll
 * @param pitch
 * @param yaw
 * @return Eigen::Quaternion<double>
 */
Eigen::Quaternion<double> RPYToQuaterion(const double roll, const double pitch,
                                         const double yaw);

}  // namespace math
}  // namespace damotion

#endif /* MATH_QUATERNION_H */
