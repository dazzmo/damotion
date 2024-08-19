#ifndef OPTIMISATION_INITIAL_VALUE_HPP
#define OPTIMISATION_INITIAL_VALUE_HPP

#include <Eigen/Core>

namespace damotion {
namespace optimisation {

template <class ObjectType>
class InitialiseableObject {
 public:
  using Type = ObjectType;

  InitialiseableObject() = default;
  InitialiseableObject(const ObjectType &initial_value)
      : initial_value_(initial_value) {}

  ObjectType &initialValue() { return initial_value_; }
  const ObjectType &initialValue() const { return initial_value_; }

 private:
  ObjectType initial_value_;
};

// Template specialisations
template <>
class InitialiseableObject<Eigen::VectorXd> {
 public:
  InitialiseableObject(const std::size_t &sz)
      : initial_value_(Eigen::VectorXd::Zero(sz)) {}

 private:
  Eigen::VectorXd initial_value_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* OPTIMISATION_INITIAL_VALUE_HPP */
