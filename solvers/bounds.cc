#include "solvers/bounds.h"
namespace damotion {
namespace optimisation {

void SetBounds(Eigen::Ref<Eigen::VectorXd> ub, Eigen::Ref<Eigen::VectorXd> lb,
               const BoundsType type) {
    switch (type) {
        case BoundsType::kEquality: {
            ub.setConstant(0.0);
            lb.setConstant(0.0);
            break;
        }

        case BoundsType::kPositive: {
            ub.setConstant(std::numeric_limits<double>::infinity());
            lb.setConstant(0.0);
            break;
        }

        case BoundsType::kNegative: {
            ub.setConstant(0.0);
            lb.setConstant(-std::numeric_limits<double>::infinity());
            break;
        }

        case BoundsType::kStrictlyPositive: {
            ub.setConstant(std::numeric_limits<double>::infinity());
            lb.setConstant(std::numeric_limits<double>::epsilon());
            break;
        }

        case BoundsType::kStrictlyNegative: {
            ub.setConstant(-std::numeric_limits<double>::epsilon());
            lb.setConstant(-std::numeric_limits<double>::infinity());
            break;
        }

        case BoundsType::kUnbounded: {
            ub.setConstant(std::numeric_limits<double>::infinity());
            lb.setConstant(-std::numeric_limits<double>::infinity());
            break;
        }

        case BoundsType::kCustom: {
            ub.setConstant(std::numeric_limits<double>::infinity());
            lb.setConstant(-std::numeric_limits<double>::infinity());
            break;
        }

        default: {
            ub.setConstant(std::numeric_limits<double>::infinity());
            lb.setConstant(-std::numeric_limits<double>::infinity());
            break;
        }
    }
}
}  // namespace optimisation
}  // namespace damotion