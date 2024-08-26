
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/casadi/function.hpp"
#include "damotion/core/logging.hpp"
#include "damotion/core/profiler.hpp"
#include "damotion/optimisation/constraints.hpp"
#include "damotion/optimisation/program.h"
#include "damotion/symbolic/variable.hpp"

using sym = ::casadi::SX;
using dm = ::casadi::DM;

namespace dcas = damotion::casadi;
namespace dsym = damotion::symbolic;
namespace dopt = damotion::optimisation;

TEST(ConstraintVector, AddConstraint) {
  sym x = sym::sym("x", 1);
  dm c = dm::rand(1);
  dm b = dm::rand(1);
  // Create symbolic constraint
  sym f = sym(c) * x + sym(b);

  // Create linear constraint function
  dopt::Constraint::SharedPtr con =
      std::make_shared<dcas::Constraint>("constraint", f, x);

  dsym::Variable z("z"), y("y");

  dsym::Vector x1(1), x2(1);
  x1 << z;
  x2 << y;

  dopt::ConstraintVector cv;
  cv.add(con, x1, {});
  EXPECT_EQ(cv.size(), 1);

  cv.add(con, x2, {});
  EXPECT_EQ(cv.size(), 2);
}

TEST(ConstraintVector, AddLinearConstraint) {
  sym x = sym::sym("x", 1);
  dm c = dm::rand(1);
  dm b = dm::rand(1);
  // Create symbolic constraint
  sym f = sym(c) * x + sym(b);

  // Create linear constraint function
  dopt::LinearConstraint::SharedPtr con =
      std::make_shared<dcas::LinearConstraint>("lc", f, x);

  dsym::Variable z("z"), y("y");

  dsym::Vector x1(1), x2(1);
  x1 << z;
  x2 << y;

  dopt::ConstraintVector cv;
  cv.add(con, x1, {});
  EXPECT_EQ(cv.size(), 1);

  cv.add(con, x2, {});
  EXPECT_EQ(cv.size(), 2);
}

TEST(ConstraintVector, ConstraintJacobian) {
  sym x = sym::sym("x", 2);
  // Create symbolic constraint
  sym c = x(0) + x(1) * x(0) + x(1);

  // Create linear constraint function
  dopt::Constraint::SharedPtr con =
      std::make_shared<dcas::Constraint>("con", c, x);

  dsym::Vector x1 = dsym::createVector("x1", 2);
  // dopt::ConstraintVector cv;
  // cv.add(con, x1, {});

  // Create variable vector
  dsym::VariableVector vv;
  vv.add(x1);

  Eigen::VectorXd val(2), lam(2);
  Eigen::MatrixXd jac(1, 2);
  lam.setRandom();
  val.setRandom();

  double v_true = val[0] + val[0] * val[1] + val[1];
  Eigen::MatrixXd jac_true(1, 2);
  jac_true << 1.0 + val[1], 1.0 + val[0];

  // Test constraint evaluation
  Eigen::VectorXd res = con->evaluate(val, jac);
  EXPECT_EQ(res[0], v_true);
  EXPECT_TRUE(jac.isApprox(jac_true));

  // Eigen::MatrixXd J = constraintJacobian(val, cv, vv);

  // std::cout << J << '\n';
  // std::cout << jac_true << '\n';

  // EXPECT_TRUE(J.isApprox(jac_true));
}

TEST(ConstraintVector, ConstraintHessian) {
  sym x = sym::sym("x", 2);
  dm c = dm::rand(2, 2);
  dm b = dm::rand(2);
  // Create symbolic constraint
  sym f = sym::mtimes(c, x) + sym(b);

  // Create linear constraint function
  dopt::LinearConstraint::SharedPtr con =
      std::make_shared<dcas::LinearConstraint>("lc", f, x);

  dsym::Variable z("z"), y("y");

  dsym::Vector x1 = dsym::createVector("x1", 2),
               x2 = dsym::createVector("x2", 2);

  dopt::ConstraintVector cv;
  cv.add(con, x1, {});
  cv.add(con, x2, {});

  // Create variable vector
  dsym::VariableVector vv;
  vv.add(x1);
  vv.add(x2);

  Eigen::VectorXd val(4), lam(4);
  lam.setRandom();
  val.setRandom();
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  int status = RUN_ALL_TESTS();
  damotion::Profiler summary;
  return status;
}