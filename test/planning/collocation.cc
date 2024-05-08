#define DAMOTION_USE_PROFILING
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/planning/optimisation/collocation/base.h"

namespace sym = damotion::symbolic;
namespace opt = damotion::optimisation;

namespace {
TEST(Collocation, Trapezoidal) {
  // Create second-order dynamic system
  casadi::SX q = casadi::SX::sym("q", 2);
  casadi::SX v = casadi::SX::sym("v", 2);
  casadi::SX u = casadi::SX::sym("u", 1);
  casadi::SX a = 1.0 * q + 2.0 * v;
  a(1) -= 10 * u;
  // Create state
  casadi::SX x = casadi::SX::vertcat({q, v}), f = casadi::SX::vertcat({v, a});

  // Create collocation constraint
  damotion::planning::optimisation::TrapezoidalCollocationConstraint con(
      4, 4, 1, f, x, u);

  auto constraint = con.GetConstraint<Eigen::MatrixXd>();
  auto constraint_sparse = con.GetConstraint<Eigen::SparseMatrix<double>>();
}

TEST(Collocation, SimpsonHermite) {
  // Create second-order dynamic system
  casadi::SX q = casadi::SX::sym("q", 2);
  casadi::SX v = casadi::SX::sym("v", 2);
  casadi::SX u = casadi::SX::sym("u", 1);
  casadi::SX a = 1.0 * q + 2.0 * v;
  a(1) -= 10 * u;
  // Create state
  casadi::SX x = casadi::SX::vertcat({q, v}), f = casadi::SX::vertcat({v, a});

  // Create collocation constraint
  damotion::planning::optimisation::HermiteSimpsonCollocationConstraint con(
      4, 4, 1, f, x, u);

  auto constraint = con.GetConstraint<Eigen::MatrixXd>();
  auto constraint_sparse = con.GetConstraint<Eigen::SparseMatrix<double>>();
}

}  // namespace

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
