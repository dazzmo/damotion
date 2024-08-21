#define DAMOTION_USE_PROFILING

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

  Eigen::VectorXd val(4);
  val.setRandom();

  Eigen::MatrixXd A = constraintJacobian(val, cv, vv);

  std::cout << A << std::endl;
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  int status = RUN_ALL_TESTS();
  damotion::Profiler summary;
  return status;
}