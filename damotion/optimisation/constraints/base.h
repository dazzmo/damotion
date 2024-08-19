#ifndef CONSTRAINTS_BASE_H
#define CONSTRAINTS_BASE_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>

#include "damotion/casadi/function.hpp"
#include "damotion/core/function.hpp"
#include "damotion/optimisation/bounds.hpp"

namespace damotion {
namespace optimisation {

/**
 * @brief Generic constraint with a single vector input
 *
 */
class Constraint : public FunctionBase<1, Eigen::VectorXd>,
                   public BoundedObject<Eigen::VectorXd> {
 public:
  using SharedPtr = std::shared_ptr<Constraint>;
  using UniquePtr = std::unique_ptr<Constraint>;

  using Base = FunctionBase<1, Eigen::VectorXd>;

  const std::string &name() const { return name_; }

  Constraint() = default;
  ~Constraint() = default;

  static bool isSatisfied(const ReturnType &c) { return true; }

 private:
  Index dim_ = 0;
  std::string name_ = "";
};

// TODO - Create constraint violation function

}  // namespace optimisation
}  // namespace damotion

#endif/* CONSTRAINTS_BASE_H */
