#define DAMOTION_USE_PROFILING
#include "damotion/solvers/ipopt.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/optimisation/program.h"

namespace sym = damotion::symbolic;
namespace opt = damotion::optimisation;

namespace {

TEST(IpoptTest, IpoptSolverCreation) {
  // Create codegen function
  sym::VariableVector x = sym::CreateVariableVector("x", 10);

  opt::SparseProgram program;

  Eigen::Matrix<double, 1, 10> A1;
  A1.setZero();

  A1[2] = 1.0;
  A1[9] = -1.0;
  Eigen::Vector<double, 1> b1(1.0);

  opt::LinearConstraint<Eigen::SparseMatrix<double>>::SharedPtr con1 =
      std::make_shared<opt::LinearConstraint<Eigen::SparseMatrix<double>>>(
          "", A1, b1, opt::Bounds::Type::kEquality);
  // Create sparse constraint by symbolic expression
  casadi::SX xs = casadi::SX::sym("x", 2);
  sym::VariableVector xcon2(2);
  xcon2 << x[5], x[9];

  sym::Expression expr = xs(0) - xs(1) + 2.0;
  expr.SetInputs({xs}, {});

  opt::LinearConstraint<Eigen::SparseMatrix<double>>::SharedPtr con2 =
      std::make_shared<opt::LinearConstraint<Eigen::SparseMatrix<double>>>(
          "", expr, opt::Bounds::Type::kEquality);

  program.addDecisionVariables(x);

  program.AddLinearConstraint(con1, {x}, {});
  program.AddLinearConstraint(con2, {xcon2}, {});

  casadi::SX xx = casadi::SX::sym("x", 10);
  sym::Expression J = dot(xx, xx) + xx(0) + xx(0) * xx(2);
  J.SetInputs({xx}, {});
  opt::QuadraticCost<Eigen::SparseMatrix<double>>::SharedPtr cost =
      std::make_shared<opt::QuadraticCost<Eigen::SparseMatrix<double>>>(
          "sum_squares", J);

  program.AddQuadraticCost(cost, {x}, {});

  // Create optimisation vector
  program.setDecisionVariableVector();

  program.AddBoundingBoxConstraint(-10.0, 10.0, x);

  program.PrintProgramSummary();

  opt::solvers::IpoptSolver solver(program);
  solver.solve();

  // Create dummy optimisation variable
  Eigen::VectorXd xopt(10);
  xopt.setOnes();

  damotion::common::Profiler summary;
}
}  // namespace

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
