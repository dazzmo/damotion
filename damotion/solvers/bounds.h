#ifndef BOUNDS_H
#define BOUNDS_H

#include <Eigen/Core>
#include <limits>

namespace damotion {
namespace optimisation {

enum class BoundsType {
  kEquality,
  kPositive,
  kNegative,
  kStrictlyPositive,
  kStrictlyNegative,
  kUnbounded,
  kCustom
};

void SetBounds(Eigen::Ref<Eigen::VectorXd> ub, Eigen::Ref<Eigen::VectorXd> lb,
               const BoundsType type = BoundsType::kUnbounded);

}  // namespace optimisation
}  // namespace damotion

#endif /* BOUNDS_H */
