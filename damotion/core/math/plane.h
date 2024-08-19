#ifndef MATH_PLANE_H
#define MATH_PLANE_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>

#include "damotion/casadi/eigen.h"
#include "damotion/casadi/function.hpp"

namespace damotion {
namespace core {

/**
 * @brief Class to represent a hyperplane in Euclidean space.
 *
 */
template <int dim = 3>
class Hyperplane {
 public:
  Hyperplane()
      : normal(Eigen::VectorX<dim>::Zero()), p(Eigen::VectorX<dim>::Zero()) {}
  ~Hyperplane() = default;

  /**
   * @brief Returns an expression for the signed symbolic distance between this
   * and the point p.
   *
   * @param p
   * @return casadi::SX
   */
  casadi::SX SymbolicDistance(const casadi::SX &point) {
    casadi::DM a, p;
    damotion::casadi::toCasadi(this->normal, a);
    damotion::casadi::toCasadi(this->p, p);
    // Get signed distance
    return (casadi::SX::dot(a, point) - casadi::SX::dot(a, p)) /
           casadi::SX::norm_2(a);
  }

  /**
   * @brief Returns the signed distance between the plane and the point p.
   *
   * @param p
   * @return double
   */
  double Distance(const Eigen::VectorXd &p) {
    return normal.dot(p - point) / normal.norm();
  }

  /**
   * @brief Returns the minimum distance between this and the provided plane
   *
   * @param p
   * @return double
   */
  double Distance(const Hyperplane &plane) { return 0.0; }

 private:
  // Point that the plane passes through
  Eigen::VectorX<dim> point;
  // Normal of the plane
  Eigen::VectorX<dim> normal;
};

}  // namespace core
}  // namespace damotion

#endif /* MATH_PLANE_H */
