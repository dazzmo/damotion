#ifndef UTILS_LOG_H
#define UTILS_LOG_H

#include <casadi/casadi.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/spatial/explog.hpp>

namespace damotion {

template <typename T>
Eigen::Matrix<T, 3, -1> log3(const Eigen::Matrix<T, 3, 3> &R) {
    T theta;
    return pinocchio::log3(R, theta);
}

template <typename T>
Eigen::Matrix<T, 3, -1> log3(const Eigen::Matrix<T, 3, 3> &R, T &theta) {
    return pinocchio::log3(R, theta);
}

/**
 * @brief Symbolic function to compute the logarithm map for a matrix R in SO3.
 * Based on pinocchio's implementation of the function
 * (pinocchio/spatial/log.hxx) which includes limits for numerical stability.
 *
 * @param R A matrix in SO3 to be mapped to the Lie Algebra of so3
 * @return Eigen::Matrix<casadi::SX, 3, -1>
 */
template <>
Eigen::Matrix<casadi::SX, 3, -1> log3(const Eigen::Matrix<casadi::SX, 3, 3> &R,
                                      casadi::SX &theta);

template <>
Eigen::Matrix<casadi::SX, 3, -1> log3(const Eigen::Matrix<casadi::SX, 3, 3> &R);

template <typename T>
void Jlog3(T &theta, const Eigen::Matrix<T, 3, 1> &log,
           Eigen::Matrix<T, 3, 3> &J) {
    return pinocchio::Jlog3(theta, log, J);
}

template <>
void Jlog3(casadi::SX &theta, const Eigen::Matrix<casadi::SX, 3, 1> &log,
           Eigen::Matrix<casadi::SX, 3, 3> &J);

template <typename T>
Eigen::Matrix<T, 6, -1> log6(const Eigen::Matrix<T, 3, 3> &R,
                             const Eigen::Matrix<T, 3, 1> &p) {
    return pinocchio::log6(pinocchio::SE3Tpl<T>(R, p));
}

/**
 * @brief Symbolic function to compute the logarithm map for a homogeneous
 * matrix T in SE3. Based on pinocchio's implementation of the function
 * (pinocchio/spatial/log.hxx) which includes limits for numerical stability.
 *
 * @param R A matrix in SO3 to be mapped to the Lie Algebra of so3
 * @return Eigen::Matrix<casadi::SX, 3, -1>
 */
template <>
Eigen::Matrix<casadi::SX, 6, -1> log6(const Eigen::Matrix<casadi::SX, 3, 3> &R,
                                      const Eigen::Matrix<casadi::SX, 3, 1> &p);

template <typename T>
void Jlog6(const Eigen::Matrix<T, 3, 3> &R, const Eigen::Matrix<T, 3, 1> &p,
           Eigen::Matrix<T, 6, 6> &J) {
    return pinocchio::Jlog6(pinocchio::SE3Tpl<T>(R, p), J);
}

template <>
void Jlog6(const Eigen::Matrix<casadi::SX, 3, 3> &R,
           const Eigen::Matrix<casadi::SX, 3, 1> &p,
           Eigen::Matrix<casadi::SX, 6, 6> &J);

}  // namespace damotion

#endif /* UTILS_LOG_H */
