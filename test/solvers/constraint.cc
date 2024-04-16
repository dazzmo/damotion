#define DAMOTION_USE_PROFILING
#include <gtest/gtest.h>

#include "solvers/constraint.h"

namespace sym = damotion::symbolic;
namespace opt = damotion::optimisation;

TEST(Constraint, ConstructByExpression) {
    // // Create quadratic cost
    // casadi::SX x = casadi::SX::sym("x", 2);
    // sym::Expression c = casadi::SX::zeros(2, 1);
    // c(0) = 2.0 * x(0) + x(1);
    // c(1) = 2.0 * x(0) * x(1);
    // c.SetInputs({x}, {});

    // opt::Constraint constraint("con", c, opt::BoundsType::kEquality);

    // // Test constraint by evaluation
    // Eigen::VectorXd xt(2);
    // xt.setRandom();

    // constraint.Update({xt}, true, false);

    // Eigen::VectorXd ctrue(2);
    // ctrue[0] = 2.0 * xt(0) + xt(1);
    // ctrue[1] = 2.0 * xt(0) * xt(1);

    // EXPECT_TRUE(ctrue.isApprox(constraint.Vector()));
}

