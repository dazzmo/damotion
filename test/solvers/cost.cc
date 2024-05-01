#define DAMOTION_USE_PROFILING
#include <gtest/gtest.h>

#include "solvers/program.h"
#include "solvers/solve_qpoases.h"

namespace sym = damotion::symbolic;
namespace opt = damotion::optimisation;

TEST(QuadraticCost, WithDecisionVariables) {
  // Create quadratic cost
  casadi::SX x = casadi::SX::sym("x", 2);
  sym::Expression J = 3.0 * dot(x, x) + 2 * (x(0) - x(1)) + 5.0;
  J.SetInputs({x}, {});

  std::shared_ptr<opt::QuadraticCost> q_cost =
      std::make_shared<opt::QuadraticCost>("test_cost", J, true, true);

  Eigen::VectorXd x_test(2);
  x_test << 1.0, 1.0;

  q_cost->ObjectiveFunction().setInput(0, x_test.data());
  q_cost->GradientFunction().setInput(0, x_test.data());
  q_cost->HessianFunction().setInput(0, x_test.data());
  q_cost->ObjectiveFunction().call();
  q_cost->GradientFunction().call();
  q_cost->HessianFunction().call();

  // Evaluate the cost objectives
  Eigen::MatrixXd Q_true(2, 2);
  Eigen::VectorXd g_true(2);
  double c_true;

  Q_true << 6.0, 0.0, 0.0, 6.0;
  g_true << 2.0, -2.0;
  c_true = 5.0;

  EXPECT_TRUE(Q_true.isApprox(q_cost->Q()));
  EXPECT_TRUE(g_true.isApprox(q_cost->g()));
  EXPECT_DOUBLE_EQ(c_true, q_cost->c());
}

TEST(QuadraticCost, WithDecisionVariablesAndParameters) {
  // Create quadratic cost
  casadi::SX x = casadi::SX::sym("x", 2);
  casadi::SX p = casadi::SX::sym("p", 1);
  sym::Expression J = 3.0 * dot(x, x) + 2 * (p(0) * x(0) - x(1)) + 5.0;
  J.SetInputs({x}, {p});

  std::shared_ptr<opt::QuadraticCost> q_cost =
      std::make_shared<opt::QuadraticCost>("test_cost", J, true, true);

  Eigen::VectorXd x_test(2), p_test(1);
  x_test << 1.0, 1.0;
  p_test << 1.0;

  q_cost->ObjectiveFunction().setInput(0, x_test.data());
  q_cost->GradientFunction().setInput(0, x_test.data());
  q_cost->HessianFunction().setInput(0, x_test.data());
  q_cost->ObjectiveFunction().setInput(1, p_test.data());
  q_cost->GradientFunction().setInput(1, p_test.data());
  q_cost->HessianFunction().setInput(1, p_test.data());

  q_cost->ObjectiveFunction().call();
  q_cost->GradientFunction().call();
  q_cost->HessianFunction().call();

  // Evaluate the cost objectives
  Eigen::MatrixXd Q_true(2, 2);
  Eigen::VectorXd g_true(2);
  double c_true;

  Q_true << 6.0, 0.0, 0.0, 6.0;
  g_true << 2.0, -2.0;
  c_true = 5.0;

  EXPECT_TRUE(Q_true.isApprox(q_cost->Q()));
  EXPECT_TRUE(g_true.isApprox(q_cost->g()));
  EXPECT_DOUBLE_EQ(c_true, q_cost->c());

  p_test << -1.0;
  q_cost->ObjectiveFunction().call();
  q_cost->GradientFunction().call();
  q_cost->HessianFunction().call();

  g_true << -2.0, -2.0;

  EXPECT_TRUE(Q_true.isApprox(q_cost->Q()));
  EXPECT_TRUE(g_true.isApprox(q_cost->g()));
  EXPECT_DOUBLE_EQ(c_true, q_cost->c());
}
