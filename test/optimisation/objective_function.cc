
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/casadi/function.hpp"
#include "damotion/core/logging.hpp"
#include "damotion/core/profiler.hpp"
#include "damotion/optimisation/costs.hpp"
#include "damotion/optimisation/program.h"
#include "damotion/symbolic/variable.hpp"

using sym = ::casadi::SX;
using dm = ::casadi::DM;

namespace dcas = damotion::casadi;
namespace dsym = damotion::symbolic;
namespace dopt = damotion::optimisation;

TEST(ObjectiveFunction, AddCost) {
  sym x = sym::sym("x", 1);
  dm c = dm::rand(1);
  dm b = dm::rand(1);
  // Create symbolic constraint
  sym f = sym(c) * x + sym(b);

  dopt::Cost::SharedPtr obj = std::make_shared<dcas::Cost>("cost", f, x);

  dsym::Variable z("z"), y("y");

  dsym::Vector x1(1), x2(1);
  x1 << z;
  x2 << y;

  dopt::ObjectiveFunction of;
  of.add(obj, x1, {});
  of.add(obj, x2, {});
}

TEST(ObjectiveFunction, AddLinearCost) {
  sym x = sym::sym("x", 1);
  dm c = dm::rand(1);
  dm b = dm::rand(1);
  // Create symbolic constraint
  sym f = sym(c) * x + sym(b);

  dopt::LinearCost::SharedPtr obj =
      std::make_shared<dcas::LinearCost>("cost", f, x);

  dsym::Variable z("z"), y("y");

  dsym::Vector x1(1), x2(1);
  x1 << z;
  x2 << y;

  dopt::ObjectiveFunction of;
  of.add(obj, x1, {});
  of.add(obj, x2, {});
}

TEST(ObjectiveFunction, ObjectiveHessian) {
  sym x = sym::sym("x", 2);
  // Create symbolic constraint
  sym f = x(0) * x(1) + sin(x(0) + pow(x(1), 3));

  // Create linear constraint function
  dopt::Cost::SharedPtr obj = std::make_shared<dcas::Cost>("lc", f, x);

  dsym::Variable z("z"), y("y");

  dsym::Vector x1 = dsym::createVector("x1", 2),
               x2 = dsym::createVector("x2", 2);

  dopt::ObjectiveFunction of;
  of.add(obj, x1, {});
  of.add(obj, x2, {});

  // Create variable vector
  dsym::VariableVector vv;
  vv.add(x1);
  vv.add(x2);

  dsym::Vector reordering(4);
  reordering << x1[1], x2[1], x1[0], x2[0];

  vv.reorder(reordering);

  Eigen::VectorXd val(4);
  val.setRandom();

  Eigen::MatrixXd A = objectiveHessian(val, of, vv);

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