#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/casadi/codegen.h"
#include "damotion/casadi/eigen.h"
#include "damotion/casadi/function.h"
#include "damotion/common/logging.h"

TEST(EigenWrapper, EigenWrapperLoad) {
  // Create codegen function
  casadi::SX x = casadi::SX::sym("x"), y = casadi::SX::sym("y");
  casadi::Function f("test", {x, y}, {x + y}, {"x", "y"}, {"l"});

  damotion::casadi::FunctionWrapper<double> wrapper(f);

  // Evaluate function
  Eigen::VectorXd x_in(1), y_in(1);
  x_in.setRandom();
  y_in.setRandom();
  LOG(INFO) << "Calling";
  wrapper.call({x_in, y_in});
  LOG(INFO) << "Getting Output";

  EXPECT_DOUBLE_EQ(wrapper.getOutput(0), (x_in[0] + y_in[0]));
}

TEST(EigenWrapper, ToCasadiDM) {
  // Create codegen function
  int n = 10;
  Eigen::VectorXd x(n);
  x.setRandom();

  // Convert to casadi::DM
  casadi::DM xd;
  damotion::casadi::toCasadi(x, xd);

  // Convert back to Eigen::VectorXd
  Eigen::VectorXd xt;
  damotion::casadi::toEigen(xd, xt);

  EXPECT_TRUE(x.isApprox(xt));
}

TEST(EigenWrapper, EigenWrapperSparse) {
  // Create codegen function
  casadi::SX x = casadi::SX::sym("x");
  casadi::SX y(2, 2);
  y(0, 0) = 1.0;
  y(1, 1) = 1.0;

  casadi::Function f("sparse_test", {x}, {y}, {"x"}, {"y"});

  damotion::casadi::FunctionWrapper<Eigen::SparseMatrix<double>> wrapper(
      damotion::casadi::codegen(f, "./tmp"));

  // Evaluate function
  Eigen::VectorXd x_in(1);
  x_in.setRandom();

  Eigen::Matrix2d I;
  I.setIdentity();

  wrapper.call({x_in});
  Eigen::SparseMatrix<double> res = wrapper.getOutput(0);

  EXPECT_TRUE(res.isApprox(I));
}

TEST(EigenWrapper, EigenWrapperCodegenLoad) {
  // Create codegen function
  casadi::SX x = casadi::SX::sym("x"), y = casadi::SX::sym("y");
  casadi::Function f("test", {x, y}, {x + y}, {"x", "y"}, {"l"});

  damotion::casadi::FunctionWrapper<double> wrapper(
      damotion::casadi::codegen(f, "./tmp"));

  // Evaluate function
  Eigen::VectorXd x_in(1), y_in(1);
  x_in.setRandom();
  y_in.setRandom();

  wrapper.call({x_in, y_in});

  EXPECT_DOUBLE_EQ(wrapper.getOutput(0), (x_in[0] + y_in[0]));
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
