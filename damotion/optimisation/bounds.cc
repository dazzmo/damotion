#include "damotion/optimisation/bounds.hpp"
namespace damotion {
namespace optimisation {

template <>
void BoundedObject<Eigen::VectorXd>::setBoundsFromType(const BoundType &type) {
  constexpr double inf = std::numeric_limits<Scalar>::infinity();
  constexpr double eps = std::numeric_limits<Scalar>::epsilon();

  switch (type) {
    case BoundType::EQUALITY: {
      ub.setConstant(0.0);
      lb.setConstant(0.0);
      break;
    }

    case BoundType::POSITIVE: {
      ub.setConstant(inf);
      lb.setConstant(0.0);
      break;
    }

    case BoundType::NEGATIVE: {
      ub.setConstant(0.0);
      lb.setConstant(-inf);
      break;
    }

    case BoundType::STRICTLY_POSITIVE: {
      ub.setConstant(inf);
      lb.setConstant(eps);
      break;
    }

    case BoundType::STRICTLY_NEGATIVE: {
      ub.setConstant(-eps);
      lb.setConstant(-inf);
      break;
    }

    case BoundType::UNBOUNDED: {
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

template <>
void BoundedObject<double>::setBoundsFromType(const BoundType &type) {
  constexpr double inf = std::numeric_limits<Scalar>::infinity();
  constexpr double eps = std::numeric_limits<Scalar>::epsilon();

  switch (type) {
    case BoundType::EQUALITY: {
      ub = 0.0;
      lb = 0.0;
      break;
    }

    case BoundType::POSITIVE: {
      ub = inf;
      lb = 0.0;
      break;
    }

    case BoundType::NEGATIVE: {
      ub = 0.0;
      lb = -inf;
      break;
    }

    case BoundType::STRICTLY_POSITIVE: {
      ub = inf;
      lb = eps;
      break;
    }

    case BoundType::STRICTLY_NEGATIVE: {
      ub = -eps;
      lb = -inf;
      break;
    }

    case BoundType::UNBOUNDED: {
      ub = inf;
      lb = -inf;
      break;
    }

    default: {
      ub = inf;
      lb = -inf;
      break;
    }
  }
}

}  // namespace optimisation
}  // namespace damotion
