#define DAMOTION_USE_PROFILING
#include "damotion/optimisation/program.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/solvers/qpoases.h"
#include "damotion/solvers/sparse.h"

namespace sym = damotion::symbolic;
namespace opt = damotion::optimisation;

namespace {

TEST(Program, AddVariables) {
  // Create codegen function
  sym::VariableVector x = sym::CreateVariableVector("x", 2);
  sym::VariableVector y = sym::CreateVariableVector("y", 3);

  opt::Program program;

  program.addDecisionVariables(x);
  program.addDecisionVariables(y);

  program.listDecisionVariables();

  EXPECT_TRUE(true);
}

TEST(Program, CreateOptimisationVector) {
  // Create codegen function
  sym::VariableVector x = sym::CreateVariableVector("x", 2);
  sym::VariableVector y = sym::CreateVariableVector("y", 2);
  sym::VariableVector z = sym::CreateVariableVector("z", 2);

  opt::Program program;

  program.addDecisionVariables(x);
  program.addDecisionVariables(y);
  program.addDecisionVariables(z);

  sym::VariableVector opt(program.numberOfDecisionVariables());
  opt << x, x, z;
  EXPECT_FALSE(program.setDecisionVariableVector(opt));

  opt << y, x, z;
  EXPECT_TRUE(program.setDecisionVariableVector(opt));

  // Check number of variables and parameters
  EXPECT_EQ(program.numberOfParameters(), 0);
  EXPECT_EQ(program.NumberOfConstraints(), 0);
  EXPECT_EQ(program.numberOfDecisionVariables(), 6);

  // Check indexing
  EXPECT_EQ(program.getDecisionVariableIndex(y[0]), 0);
  EXPECT_EQ(program.getDecisionVariableIndex(y[1]), 1);
  EXPECT_EQ(program.getDecisionVariableIndex(x[0]), 2);
  EXPECT_EQ(program.getDecisionVariableIndex(x[1]), 3);
}

TEST(Program, CreateParameters) {
  // Create codegen function
  opt::Program program;
  sym::ParameterVector a = sym::CreateParameterVector("a", 2);
  program.addParameters(a);

  EXPECT_EQ(a.rows(), 2);
  EXPECT_EQ(a.cols(), 1);

  Eigen::VectorXd b(2);
  b.setRandom();

  program.GetParameterRef(a) = b;

  EXPECT_TRUE(program.GetParameterRef(a).isApprox(b));
}

TEST(Program, AddLinearConstraint) {
  // Create codegen function
  sym::VariableVector x = sym::CreateVariableVector("x", 2);
  sym::VariableVector y = sym::CreateVariableVector("y", 2);

  opt::Program program;

  // Create constraint x0 + 2 y1 + 3 = 0
  Eigen::Matrix<double, 1, 2> A;
  A << 1.0, 2.0;
  Eigen::Vector<double, 1> b(3.0);

  std::shared_ptr<opt::LinearConstraint<Eigen::MatrixXd>> con =
      std::make_shared<opt::LinearConstraint<Eigen::MatrixXd>>(
          "", A, b, opt::Bounds::Type::kEquality);

  program.addDecisionVariables(x);
  program.addDecisionVariables(y);

  sym::VariableVector xy(2);
  xy << x[0], y[1];

  program.AddLinearConstraint(con, {xy}, {});
  LOG(INFO) << "Added Linear Constraint";

  sym::ParameterVector a = sym::CreateParameterVector("a", 2);
  program.addParameters(a);
  Eigen::Map<Eigen::VectorXd> aref = program.GetParameterRef(a);

  LOG(INFO) << "Added Parameter";

  // Create random cost
  casadi::SX xx = casadi::SX::sym("x", 4);
  sym::Expression J = dot(xx, xx) + xx(0) + xx(0) * xx(2);
  J.SetInputs({xx}, {});
  opt::QuadraticCost<Eigen::MatrixXd>::SharedPtr cost =
      std::make_shared<opt::QuadraticCost<Eigen::MatrixXd>>("sum_squares", J);

  sym::VariableVector xxyy(4);
  xxyy << x, y;
  program.AddQuadraticCost(cost, {xxyy}, {});

  LOG(INFO) << "Added Quadratic Cost";

  // Create optimisation vector
  program.setDecisionVariableVector();

  program.AddBoundingBoxConstraint(-1.0, 1.0, x);
  program.AddBoundingBoxConstraint(-2.0, 2.0, y);

  LOG(INFO) << "Added Bounding Box Constraints";

  program.PrintProgramSummary();

  // Create QPOASES solver and test if constraint jacobian gets created
  opt::solvers::QPOASESSolverInstance solver(program);

  LOG(INFO) << "Created solver";

  solver.Solve();
  solver.Solve();

  LOG(INFO) << solver.GetPrimalSolution();

  damotion::common::Profiler summary;
}

TEST(Program, SparseProgram) {
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

  auto binding1 = program.AddLinearConstraint(con1, {x}, {});
  auto binding2 = program.AddLinearConstraint(con2, {xcon2}, {});

  // Print constraint data
  LOG(INFO) << "Constraint 1";
  LOG(INFO) << "A = " << con1->A();
  LOG(INFO) << "b = " << con1->b();

  LOG(INFO) << "Constraint 2";
  LOG(INFO) << "A = " << con2->A();
  LOG(INFO) << "b = " << con2->b();

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
}
}  // namespace

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
