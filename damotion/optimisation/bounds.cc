#include "damotion/optimisation/bounds.hpp"
namespace damotion {
namespace optimisation {

void BoundedObject<Eigen::VectorXd>::setBoundsFromType(const BoundType &type) {
  constexpr double inf = std::numeric_limits<double>::infinity();
  constexpr double eps = std::numeric_limits<double>::epsilon();

  switch (type) {
    case BoundType::EQUALITY: {
      ub_.setConstant(0.0);
      lb_.setConstant(0.0);
      break;
    }

    case BoundType::POSITIVE: {
      ub_.setConstant(inf);
      lb_.setConstant(0.0);
      break;
    }

    case BoundType::NEGATIVE: {
      ub_.setConstant(0.0);
      lb_.setConstant(-inf);
      break;
    }

    case BoundType::STRICTLY_POSITIVE: {
      ub_.setConstant(inf);
      lb_.setConstant(eps);
      break;
    }

    case BoundType::STRICTLY_NEGATIVE: {
      ub_.setConstant(-eps);
      lb_.setConstant(-inf);
      break;
    }

    case BoundType::UNBOUNDED: {
      ub_.setConstant(inf);
      lb_.setConstant(-inf);
      break;
    }

    default: {
      ub_.setConstant(inf);
      lb_.setConstant(-inf);
      break;
    }
  }
}

void BoundedObject<double>::setBoundsFromType(const BoundType &type) {
  constexpr double inf = std::numeric_limits<double>::infinity();
  constexpr double eps = std::numeric_limits<double>::epsilon();

  switch (type) {
    case BoundType::EQUALITY: {
      ub_ = 0.0;
      lb_ = 0.0;
      break;
    }

    case BoundType::POSITIVE: {
      ub_ = inf;
      lb_ = 0.0;
      break;
    }

    case BoundType::NEGATIVE: {
      ub_ = 0.0;
      lb_ = -inf;
      break;
    }

    case BoundType::STRICTLY_POSITIVE: {
      ub_ = inf;
      lb_ = eps;
      break;
    }

    case BoundType::STRICTLY_NEGATIVE: {
      ub_ = -eps;
      lb_ = -inf;
      break;
    }

    case BoundType::UNBOUNDED: {
      ub_ = inf;
      lb_ = -inf;
      break;
    }

    default: {
      ub_ = inf;
      lb_ = -inf;
      break;
    }
  }
}

}  // namespace optimisation
}  // namespace damotion
