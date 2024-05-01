#include "symbolic/expression.h"

#include <gtest/gtest.h>

TEST(Expression, Expression) {
  casadi::SX x = casadi::SX::sym("x", 2);
  casadi::SX y = casadi::SX::sym("y", 2);

  damotion::symbolic::Expression a = x;
  damotion::symbolic::Expression b = dot(x, y);

  damotion::symbolic::Expression c = a + b;

  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;
  std::cout << x + dot(x, y) << std::endl;

  b.SetInputs({x, y}, {});

  EXPECT_TRUE(casadi::SX::is_equal(a, x));
  EXPECT_TRUE(casadi::SX::is_equal(b, dot(x, y)));
  EXPECT_TRUE(b.Variables().size() == 2);
  EXPECT_TRUE(b.Parameters().size() == 0);
  EXPECT_TRUE(casadi::SX::is_equal(c, x + dot(x, y)));
}

TEST(Expression, ExpressionVector) {
  // Create codegen function

  EXPECT_TRUE(false);
}
