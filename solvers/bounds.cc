#include "solvers/bounds.h"
// namespace damotion {
// namespace optimisation {

// void SetBounds(Eigen::Ref<VectorXd> ub, Eigen::Ref<VectorXd> lb,
//                const Bounds bounds) {
//     switch (bounds) {
//         case Bounds::kEquality: {
//             ub.setConstant(0.0);
//             lb.setConstant(0.0);
//         }

//         case Bounds::kPositive: {
//             ub.setConstant(std::numeric_limits<double>::infinity());
//             lb.setConstant(0.0);
//         }

//         case Bounds::kNegative: {
//             ub.setConstant(0.0);
//             lb.setConstant(-std::numeric_limits<double>::infinity());
//         }

//         case Bounds::kStrictlyPositive: {
//             ub.setConstant(std::numeric_limits<double>::infinity());
//             lb.setConstant(std::numeric_limits<double>::epsilon());
//         }

//         case Bounds::kStrictlyNegative: {
//             ub.setConstant(-std::numeric_limits<double>::epsilon());
//             lb.setConstant(-std::numeric_limits<double>::infinity());
//         }

//         case Bounds::kUnbounded: {
//             ub.setConstant(std::numeric_limits<double>::infinity());
//             lb.setConstant(-std::numeric_limits<double>::infinity());
//         }

//         case Bounds::kCustom: {
//             ub.setConstant(std::numeric_limits<double>::infinity());
//             lb.setConstant(-std::numeric_limits<double>::infinity());
//         }

//         default: {
//             ub.setConstant(std::numeric_limits<double>::infinity());
//             lb.setConstant(-std::numeric_limits<double>::infinity());
//         }
//     }
// }
// }  // namespace optimisation
// }  // namespace damotion