#include "utils/casadi.h"

#include <gtest/gtest.h>

TEST(CasadiUtils, CreateGradientFunction) {
  // Create codegen function
  casadi::SX x = casadi::SX::sym("x"), y = casadi::SX::sym("y");

  casadi::SX f = x * y + (x - y);

  casadi::Function fx = damotion::utils::casadi::CreateGradientFunction(
      "f", f, {x, y}, {"x", "y"}, {x, y}, {"x", "y"});

  std::cout << fx << std::endl;

  EXPECT_TRUE(false);
}

TEST(CasadiUtils, CreateHessianFunction) {
  // Create codegen function
  casadi::SX x = casadi::SX::sym("x"), y = casadi::SX::sym("y");

  casadi::SX f = x * y + (x - y);

  casadi::Function fx = damotion::utils::casadi::CreateHessianFunction(
      "f", f, {x, y}, {"x", "y"}, {{x, x}, {y, y}, {x, y}},
      {{"x", "x"}, {"y", "y"}, {"x", "y"}});

  std::cout << fx << std::endl;

  EXPECT_TRUE(false);

  // Create codegen function
  x = casadi::SX::sym("x", 2);
  y = casadi::SX::sym("y", 3);

  f = x(1) * y(2) * y(1) + sin(x(0)) + cos(y(0) * x(1));

  casadi::Function fx_2 = damotion::utils::casadi::CreateHessianFunction(
      "f", f, {x, y}, {"x", "y"}, {{x, x}, {y, y}, {x, y}},
      {{"x", "x"}, {"y", "y"}, {"x", "y"}});

  std::cout << fx_2 << std::endl;

  EXPECT_TRUE(false);
}
