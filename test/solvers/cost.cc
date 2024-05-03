#define DAMOTION_USE_PROFILING
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/optimisation/program.h"
#include "damotion/solvers/qpoases.h"

namespace sym = damotion::symbolic;
namespace opt = damotion::optimisation;

TEST(QuadraticCost, WithDecisionVariables) {
  // Create quadratic cost
  casadi::SX x = casadi::SX::sym("x", 2);
  sym::Expression J = 3.0 * dot(x, x) + 2 * (x(0) - x(1)) + 5.0;
  J.SetInputs({x}, {});

  opt::QuadraticCost<Eigen::MatrixXd>::SharedPtr q_cost =
      std::make_shared<opt::QuadraticCost<Eigen::MatrixXd>>("test_cost", J,
                                                            true, true);

  Eigen::VectorXd x_test(2);
  x_test << 1.0, 1.0;

  q_cost->eval({x_test}, {});
  q_cost->eval_hessian({x_test}, {});

  // Evaluate the cost objectives
  Eigen::MatrixXd Q_true(2, 2);
  Eigen::VectorXd g_true(2);
  double c_true;

  Q_true << 6.0, 0.0, 0.0, 6.0;
  g_true << 2.0, -2.0;
  c_true = 5.0;

  EXPECT_TRUE(Q_true.isApprox(q_cost->A()));
  EXPECT_TRUE(g_true.isApprox(q_cost->b()));
  EXPECT_DOUBLE_EQ(c_true, q_cost->c());
}

TEST(QuadraticCost, WithDecisionVariablesAndParameters) {
  // Create quadratic cost
  casadi::SX x = casadi::SX::sym("x", 2);
  casadi::SX p = casadi::SX::sym("p", 1);
  sym::Expression J = 3.0 * dot(x, x) + 2 * (p(0) * x(0) - x(1)) + 5.0;
  J.SetInputs({x}, {p});

  opt::QuadraticCost<Eigen::MatrixXd>::SharedPtr q_cost =
      std::make_shared<opt::QuadraticCost<Eigen::MatrixXd>>("test_cost", J,
                                                            true, true);

  Eigen::VectorXd x_test(2), p_test(1);
  x_test << 1.0, 1.0;
  p_test << 1.0;

  q_cost->eval({x_test}, {p_test});
  q_cost->eval_hessian({x_test}, {p_test});

  // Evaluate the cost objectives
  Eigen::MatrixXd Q_true(2, 2);
  Eigen::VectorXd g_true(2);
  double c_true;

  Q_true << 6.0, 0.0, 0.0, 6.0;
  g_true << 2.0, -2.0;
  c_true = 5.0;

  EXPECT_TRUE(Q_true.isApprox(q_cost->A()));
  EXPECT_TRUE(g_true.isApprox(q_cost->b()));
  EXPECT_DOUBLE_EQ(c_true, q_cost->c());

  p_test << -1.0;
  q_cost->eval({x_test}, {p_test});
  q_cost->eval_hessian({x_test}, {p_test});

  g_true << -2.0, -2.0;

  EXPECT_TRUE(Q_true.isApprox(q_cost->A()));
  EXPECT_TRUE(g_true.isApprox(q_cost->b()));
  EXPECT_DOUBLE_EQ(c_true, q_cost->c());
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
