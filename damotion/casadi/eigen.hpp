#ifndef UTILS_EIGEN_WRAPPER_H
#define UTILS_EIGEN_WRAPPER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pinocchio/autodiff/casadi.hpp>

namespace damotion {
namespace casadi {

/**
 * @brief Convert a casadi::Matrix<T> matric to an Eigen::Matrix<T>
 *
 * @tparam T
 * @tparam rows
 * @tparam cols
 * @param C
 * @param E
 */
template <typename T, int rows, int cols>
void toEigen(const ::casadi::Matrix<T> &C, Eigen::Matrix<T, rows, cols> &E) {
  E.setZero(C.rows(), C.columns());
  for (casadi_int i = 0; i < C.rows(); ++i) {
    for (casadi_int j = 0; j < C.columns(); ++j) {
      E(i, j) = T(C(i, j));
    }
  }
}

/**
 * @brief Convert an Eigen::Matrix<T> object to a casadi::Matrix<T> object
 *
 * @tparam T
 * @tparam rows
 * @tparam cols
 * @param E
 * @param C
 */
template <typename T, int rows, int cols>
void toCasadi(const Eigen::Matrix<T, rows, cols> &E, ::casadi::Matrix<T> &C) {
  C.resize(E.rows(), E.cols());
  for (int i = 0; i < E.rows(); ++i) {
    for (int j = 0; j < E.cols(); ++j) {
      // Only fill in non-zero entries
      if (!::casadi::is_zero(E(i, j))) {
        C(i, j) = E(i, j);
      }
    }
  }
}

/**
 * @brief Convert a casadi::Matrix<T> object to an
 * Eigen::Matrix<casadi::Matrix<T>> object (e.g. convert a casadi::SX to an
 * Eigen::Matrix<casadi::SX>)
 *
 * @tparam T
 * @tparam rows
 * @tparam cols
 * @param C
 * @param E
 */
template <typename T, int rows, int cols>
void toEigen(const ::casadi::Matrix<T> &C,
             Eigen::Matrix<::casadi::Matrix<T>, rows, cols> &E) {
  E.setZero(C.rows(), C.columns());
  for (casadi_int i = 0; i < C.rows(); ++i) {
    for (casadi_int j = 0; j < C.columns(); ++j) {
      E(i, j) = ::casadi::Matrix<T>(C(i, j));
    }
  }
}

template <typename T, int rows, int cols>
void toCasadi(const Eigen::Matrix<::casadi::Matrix<T>, rows, cols> &E,
              ::casadi::Matrix<T> &C) {
  C.resize(E.rows(), E.cols());
  for (int i = 0; i < E.rows(); ++i) {
    for (int j = 0; j < E.cols(); ++j) {
      // Only fill in non-zero entries
      if (!::casadi::is_zero(E(i, j)->at(0))) {
        C(i, j) = E(i, j)->at(0);
      }
    }
  }
}

}  // namespace casadi
}  // namespace damotion

#endif /* UTILS_EIGEN_WRAPPER_H */
