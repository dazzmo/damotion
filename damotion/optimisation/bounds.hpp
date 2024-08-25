/**
 * @file bounds.h
 * @author Damian Abood (damian.abood@sydney.edu.au)
 * @brief Bound types used for conventional constrained optimisation
 * @version 0.1
 * @date 2024-05-09
 *
 *
 */
#ifndef OPTIMISATION_BOUNDS_HPP
#define OPTIMISATION_BOUNDS_HPP

#include <Eigen/Core>
#include <limits>

#include "damotion/core/fwd.hpp"

namespace damotion {
namespace optimisation {

enum class BoundType {
  /**
   * @brief Bounds of the form x = 0
   *
   */
  EQUALITY,
  /**
   * @brief Bounds of the form x >= 0
   *
   */
  POSITIVE,
  /**
   * @brief Bounds of the form x <= 0
   *
   */
  NEGATIVE,
  /**
   * @brief Bounds of the form x > 0
   *
   */
  STRICTLY_POSITIVE,
  /**
   * @brief Bounds of the form x < 0
   *
   */
  STRICTLY_NEGATIVE,
  /**
   * @brief Bounds of the form -inf < x < inf
   *
   */
  UNBOUNDED,
  /**
   * @brief Custom defined upper and lower bounds
   *
   */
  CUSTOM
};

template <class ObjectType>
class BoundedObject {
 public:
  BoundedObject() = default;

  /**
   * @brief Lower bound for the object
   *
   * @return const ObjectType&
   */
  const ObjectType &lb() const { return lb_; }

  /**
   * @brief Upper bound for the object
   *
   * @return const ObjectType&
   */
  const ObjectType &ub() const { return ub_; }

 private:
  ObjectType lb_;
  ObjectType ub_;
};

template <>
class BoundedObject<double> {
 public:
  BoundedObject() {}
  BoundedObject(const BoundType &type) : lb_(), ub_() {}

  void setLowerBound(const double &lb) { lb_ = lb; }
  void setUpperBound(const double &ub) { ub_ = ub; }

  /**
   * @brief Set bounds to generic type
   *
   * @param type
   */
  void setBoundsFromType(const BoundType &type);

 private:
  double lb_;
  double ub_;
};

template <>
class BoundedObject<Eigen::VectorXd> {
 public:
  BoundedObject() : sz_(0) {}
  BoundedObject(const Index &sz, const BoundType &type = BoundType::CUSTOM)
      : sz_(sz), type_(type) {
    lb_.resize(sz);
    ub_.resize(sz);
    setBoundsFromType(type);
  }

  const Index &size() const { return sz_; }

  const BoundType &getBoundsType() const { return type_; }

  /**
   * @brief Lower bound for the object
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &lb() const { return lb_; }

  /**
   * @brief Upper bound for the object
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &ub() const { return ub_; }

  void setLowerBound(const Eigen::Ref<const Eigen::VectorXd> &lb) { lb_ = lb; }
  void setLowerBound(const double &lb) { lb_.setConstant(lb); }

  void setUpperBound(const Eigen::Ref<const Eigen::VectorXd> &ub) { ub_ = ub; }
  void setUpperBound(const double &ub) { ub_.setConstant(ub); }

  /**
   * @brief Set bounds to generic type
   *
   * @param type
   */
  void setBoundsFromType(const BoundType &type);

 protected:
  // /**
  //  * @brief Lower bound for the object
  //  *
  //  * @return  Eigen::VectorXd&
  //  */
  // Eigen::VectorXd &lb() { return lb_; }

  // /**
  //  * @brief Upper bound for the object
  //  *
  //  * @return  Eigen::VectorXd&
  //  */
  // Eigen::VectorXd &ub() { return ub_; }

 private:
  std::size_t sz_;

  BoundType type_;

  Eigen::VectorXd lb_;
  Eigen::VectorXd ub_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* OPTIMISATION_BOUNDS_HPP */
