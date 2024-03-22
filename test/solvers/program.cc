#include "solvers/program.h"

#include <gtest/gtest.h>

TEST(Program, GetNamesFromSXVector) {
    // Create codegen function
    casadi::SX x = casadi::SX::sym("x", 2), y = casadi::SX::sym("y", 3);

    casadi::StringVector inames =
        damotion::solvers::GetSXVectorNames(casadi::SXVector({x, y}));

    EXPECT_EQ(inames[0], "x");
    EXPECT_EQ(inames[1], "y");
}