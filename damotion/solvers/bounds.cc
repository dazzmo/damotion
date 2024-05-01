#include "damotion/solvers/bounds.h"

namespace damotion {
namespace optimisation {

void SetBounds(Eigen::Ref<Eigen::VectorXd> ub, Eigen::Ref<Eigen::VectorXd> lb,
               const BoundsType type) {
    const double inf = 1e19;
    const double eps = 1e-8;

    switch (type) {
        case BoundsType::kEquality: {
            ub.setConstant(0.0);
            lb.setConstant(0.0);
            break;
        }

        case BoundsType::kPositive: {
            ub.setConstant(inf);
            lb.setConstant(0.0);
            break;
        }

        case BoundsType::kNegative: {
            ub.setConstant(0.0);
            lb.setConstant(-inf);
            break;
        }

        case BoundsType::kStrictlyPositive: {
            ub.setConstant(inf);
            lb.setConstant(eps);
            break;
        }

        case BoundsType::kStrictlyNegative: {
            ub.setConstant(-eps);
            lb.setConstant(-inf);
            break;
        }

        case BoundsType::kUnbounded: {
            ub.setConstant(inf);
            lb.setConstant(-inf);
            break;
        }

        case BoundsType::kCustom: {
            ub.setConstant(inf);
            lb.setConstant(-inf);
            break;
        }

        default: {
            ub.setConstant(inf);
            lb.setConstant(-inf);
            break;
        }
    }
}
}  // namespace optimisation
}  // namespace damotion