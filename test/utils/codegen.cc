#include "utils/codegen.h"

#include <gtest/gtest.h>

TEST(Codegen, Load) {
    // Create codegen function
    casadi::SX x = casadi::SX::sym("x"), y = casadi::SX::sym("y");
    casadi::Function f("test", {x, y}, {x + y}, {"x", "y"}, {"l"});

    f = damotion::utils::casadi::codegen(f, "./");

    // Evaluate function
    casadi::DMVector in(2);
    in[0] = {1.0};
    in[1] = {1.0};
    casadi::DMVector out = f(in);

    EXPECT_DOUBLE_EQ(out[0]->at(0), 2.0);
}

TEST(Codegen, LoadDifferentFolder) {
    // Create codegen function
    casadi::SX x = casadi::SX::sym("x"), y = casadi::SX::sym("y");
    casadi::Function f("test", {x, y}, {x + y}, {"x", "y"}, {"l"});

    f = damotion::utils::casadi::codegen(f, "./tmp/");

    // Evaluate function
    casadi::DMVector in(2);
    in[0] = {1.0};
    in[1] = {1.0};
    casadi::DMVector out = f(in);

    EXPECT_DOUBLE_EQ(out[0]->at(0), 2.0);
}