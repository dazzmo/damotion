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

/**
 * @brief Sets the bounds ub and lb based on the BoundsType provided.
 * 
 * @param ub 
 * @param lb 
 * @param type 
 */
void SetBoundsByType(Eigen::Ref<Eigen::VectorXd> ub,
                     Eigen::Ref<Eigen::VectorXd> lb,
                     const BoundsType type = BoundsType::kUnbounded);

}  // namespace optimisation
}  // namespace damotion

#endif /* BOUNDS_H */
