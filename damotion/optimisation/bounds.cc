#include "damotion/optimisation/bounds.h"
namespace damotion {
namespace optimisation {

void setBoundsByType(Eigen::Ref<Eigen::VectorXd> ub,
                     Eigen::Ref<Eigen::VectorXd> lb, const Bounds::Type type) {
  const double inf = 1e19;
  const double eps = 1e-8;

  switch (type) {
    case Bounds::Type::kEquality: {
      ub.setConstant(0.0);
      lb.setConstant(0.0);
      break;
    }

    case Bounds::Type::kPositive: {
      ub.setConstant(inf);
      lb.setConstant(0.0);
      break;
    }

    case Bounds::Type::kNegative: {
      ub.setConstant(0.0);
      lb.setConstant(-inf);
      break;
    }

    case Bounds::Type::kStrictlyPositive: {
      ub.setConstant(inf);
      lb.setConstant(eps);
      break;
    }

    case Bounds::Type::kStrictlyNegative: {
      ub.setConstant(-eps);
      lb.setConstant(-inf);
      break;
    }

    case Bounds::Type::kUnbounded: {
      ub.setConstant(inf);
      lb.setConstant(-inf);
      break;
    }

    case Bounds::Type::kCustom: {
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
