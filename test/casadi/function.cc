
#include "damotion/casadi/function.hpp"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/casadi/codegen.hpp"
#include "damotion/casadi/eigen.hpp"
#include "damotion/core/logging.hpp"
#include "damotion/core/profiler.hpp"

using sym = ::casadi::SX;
using dm = ::casadi::DM;

TEST(CasadiFunction, FunctionWrapperScalar) {
  std::size_t n = 5;
  sym x = sym::sym("x", n);
  // Create symbolic constraint
  sym ex = sym::dot(x, x);

  auto fun = ::casadi::Function("function", {x}, {ex});

  // Create function wrapper
  damotion::casadi::FunctionWrapper<1, 1> fw(fun);

  Eigen::VectorXd xe(n);
  xe.setRandom();

  // Call function normally
  ::casadi::DM xc;
  damotion::casadi::toCasadi(xe, xc);
  EXPECT_DOUBLE_EQ(xe.dot(xe), fun(xc)[0]->at(0));

  double res = 0.0;

  for (std::size_t i = 0; i < 1; ++i) {
    damotion::Profiler profiler("evaluate");
    fw.call({xe}, {res});
  }

  EXPECT_DOUBLE_EQ(xe.dot(xe), res);
}

TEST(CasadiFunction, FunctionWrapperVector) {
  std::size_t n = 5;
  sym x = sym::sym("x", n);
  // Create symbolic constraint
  sym ex = x;

  auto fun = ::casadi::Function("function", {x}, {ex});

  // Create function wrapper
  damotion::casadi::FunctionWrapper<1, 1> fw(
      ::casadi::Function("function", {x}, {ex}));

  Eigen::VectorXd xe(n), res(n);
  xe.setRandom();

  for (std::size_t i = 0; i < 1; ++i) {
    damotion::Profiler profiler("evaluate");
    fw.call({xe}, {res});
  }

  EXPECT_TRUE(xe.isApprox(res));

  ::casadi::DM xc;
  damotion::casadi::toCasadi(xe, xc);
  // Also call the function normally
  Eigen::VectorXd rese;
  damotion::casadi::toEigen(fun(xc)[0], rese);
  EXPECT_TRUE(xe.isApprox(rese));
}

TEST(CasadiFunction, LinearConstraint) {
  std::size_t n = 100;
  sym x = sym::sym("x", n);
  dm A = dm::rand(n, n);
  dm b = dm::rand(n);
  // Create symbolic constraint
  sym c = sym::mtimes(A, x) + sym(b);

  // Create linear constraint function
  damotion::optimisation::LinearConstraint::SharedPtr p =
      std::make_shared<damotion::casadi::LinearConstraint>("lc", c, x);

  Eigen::VectorXd xe(n);
  xe.setRandom();

  Eigen::VectorXd res(n);

  for (std::size_t i = 0; i < 1; ++i) {
    damotion::Profiler profiler("evaluate");
    res = p->evaluate(xe);
  }

  // Convert to Eigen
  Eigen::MatrixXd Ae;
  Eigen::VectorXd be;

  damotion::casadi::toEigen(A, Ae);
  damotion::casadi::toEigen(b, be);

  EXPECT_TRUE((Ae * xe + be).isApprox(res));
}

TEST(CasadiFunction, LinearCost) {
  std::size_t n = 100;
  sym x = sym::sym("x", n);
  dm c = dm::rand(n);
  dm b = dm::rand(1);
  // Create symbolic constraint
  sym f = sym::mtimes(c.T(), x) + sym(b);

  // Create linear constraint function
  damotion::optimisation::LinearCost::SharedPtr p =
      std::make_shared<damotion::casadi::LinearCost>("lc", f, x);

  Eigen::VectorXd xe(n);
  xe.setRandom();

  double res = 0.0;

  for (std::size_t i = 0; i < 1; ++i) {
    damotion::Profiler profiler("LinearCost.evaluate");
    res = p->evaluate(xe);
  }

  // Convert to Eigen
  Eigen::VectorXd ce;
  double be;

  damotion::casadi::toEigen(c, ce);
  be = b(0)->at(0);

  double fc = ce.dot(xe) + be;

  EXPECT_EQ(fc, res);
}

TEST(CasadiFunction, QuadraticCost) {
  std::size_t n = 100;
  sym x = sym::sym("x", n);
  dm A = dm::rand(n, n);
  dm b = dm::rand(n);
  dm c = dm::rand(1);

  // Create symbolic constraint
  sym f = 0.5 * sym::mtimes(x.T(), sym::mtimes(A, x)) + sym::mtimes(b.T(), x) +
          sym(c);

  // Create linear constraint function
  damotion::optimisation::QuadraticCost::SharedPtr p =
      std::make_shared<damotion::casadi::QuadraticCost>("qc", f, x);

  Eigen::VectorXd xe(n);
  xe.setRandom();

  double res = 0.0;

  for (std::size_t i = 0; i < 1; ++i) {
    damotion::Profiler profiler("QuadraticCost.evaluate");
    res = p->evaluate(xe);
  }

  // Convert to Eigen
  Eigen::MatrixXd Ae;
  Eigen::VectorXd be;
  double ce;

  damotion::casadi::toEigen(A, Ae);
  damotion::casadi::toEigen(b, be);
  ce = c(0)->at(0);

  double fc = 0.5 * xe.dot(Ae * xe) + be.dot(xe) + ce;

  EXPECT_DOUBLE_EQ(fc, res);
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  int status = RUN_ALL_TESTS();
  damotion::Profiler summary;
  return status;
}