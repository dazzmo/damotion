#define DAMOTION_USE_PROFILING
#include "damotion/optimisation/block.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/optimisation/binding.h"
#include "damotion/optimisation/block.h"
#include "damotion/optimisation/constraints/constraints.h"
#include "damotion/optimisation/costs/costs.h"

namespace sym = damotion::symbolic;
namespace opt = damotion::optimisation;

namespace {

TEST(Program, SparseProgram) {
  typedef Eigen::SparseMatrix<double> MatType;
  // Create codegen function
  sym::VariableVector x = sym::CreateVariableVector("x", 10);

  Eigen::Matrix<double, 1, 10> A1;
  A1.setZero();

  A1[2] = 1.0;
  A1[9] = -1.0;
  Eigen::Vector<double, 1> b1(1.0);

  opt::LinearConstraint<MatType>::SharedPtr con1 =
      std::make_shared<opt::LinearConstraint<MatType>>(
          "", A1, b1, opt::Bounds::Type::kEquality);
  // Create sparse constraint by symbolic expression
  casadi::SX xs = casadi::SX::sym("x", 2);
  sym::VariableVector xcon2(2);
  xcon2 << x[5], x[9];

  sym::Expression expr = xs(0) - xs(1) + 2.0;
  expr.SetInputs({xs}, {});

  opt::LinearConstraint<MatType>::SharedPtr con2 =
      std::make_shared<opt::LinearConstraint<MatType>>(
          "", expr, opt::Bounds::Type::kEquality);

  // Create bindings
  auto binding1 = opt::Binding<opt::LinearConstraint<MatType>>(con1, {x}, {});
  auto binding2 =
      opt::Binding<opt::LinearConstraint<MatType>>(con2, {xcon2}, {});

  opt::SparseProgram program;
  program.addDecisionVariables(x);
  program.setDecisionVariableVector();

  // Create matrix for this constraints
  opt::BlockMatrixFunction block(2, 10,
                                 opt::BlockMatrixFunction::Type::kJacobian);
  block.AddBinding(binding1, program);
  block.AddBinding(binding2, program);

  block.GenerateMatrix();
}
}  // namespace

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
