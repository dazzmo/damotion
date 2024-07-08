/**
 * @file bounds.h
 * @author Damian Abood (damian.abood@sydney.edu.au)
 * @brief Bound types used for conventional constrained optimisation
 * @version 0.1
 * @date 2024-05-09
 *
 *
 */
#ifndef BOUNDS_H
#define BOUNDS_H

#include <Eigen/Core>
#include <limits>

namespace damotion {
namespace optimisation {

class Bounds {
 public:
  enum class Type {
    kEquality,
    kPositive,
    kNegative,
    kStrictlyPositive,
    kStrictlyNegative,
    kUnbounded,
    kCustom
  };
};

/**
 * @brief Sets the bounds ub and lb based on the Bounds::Type provided.
 *
 * @param ub
 * @param lb
 * @param type
 */
void setBoundsByType(Eigen::Ref<Eigen::VectorXd> ub,
                     Eigen::Ref<Eigen::VectorXd> lb,
                     const Bounds::Type type = Bounds::Type::kUnbounded);

}  // namespace optimisation
}  // namespace damotion

#endif /* BOUNDS_H */
