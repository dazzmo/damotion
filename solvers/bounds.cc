#include "solvers/bounds.h"
namespace damotion {
namespace optimisation {

void SetBounds(Eigen::Ref<Eigen::VectorXd> ub, Eigen::Ref<Eigen::VectorXd> lb,
               const BoundsType type) {
    switch (type) {
        case BoundsType::kEquality: {
            ub.setConstant(0.0);
            lb.setConstant(0.0);
        }

        case BoundsType::kPositive: {
            ub.setConstant(std::numeric_limits<double>::infinity());
            lb.setConstant(0.0);
        }

        case BoundsType::kNegative: {
            ub.setConstant(0.0);
            lb.setConstant(-std::numeric_limits<double>::infinity());
        }

        case BoundsType::kStrictlyPositive: {
            ub.setConstant(std::numeric_limits<double>::infinity());
            lb.setConstant(std::numeric_limits<double>::epsilon());
        }

        case BoundsType::kStrictlyNegative: {
            ub.setConstant(-std::numeric_limits<double>::epsilon());
            lb.setConstant(-std::numeric_limits<double>::infinity());
        }

        case BoundsType::kUnbounded: {
            ub.setConstant(std::numeric_limits<double>::infinity());
            lb.setConstant(-std::numeric_limits<double>::infinity());
        }

        case BoundsType::kCustom: {
            ub.setConstant(std::numeric_limits<double>::infinity());
            lb.setConstant(-std::numeric_limits<double>::infinity());
        }

        default: {
            ub.setConstant(std::numeric_limits<double>::infinity());
            lb.setConstant(-std::numeric_limits<double>::infinity());
        }
    }
}
}  // namespace optimisation
}  // namespace damotion