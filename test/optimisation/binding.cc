
#include "damotion/optimisation/binding.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/casadi/function.hpp"
#include "damotion/core/logging.hpp"
#include "damotion/core/profiler.hpp"
#include "damotion/optimisation/costs.hpp"
#include "damotion/symbolic/variable.hpp"

using sym = ::casadi::SX;
using dm = ::casadi::DM;

namespace dcas = damotion::casadi;
namespace dsym = damotion::symbolic;
namespace dopt = damotion::optimisation;

TEST(Optimisation, BindScalarCost) {
  sym x = sym::sym("x", 1);
  dm c = dm::rand(1);
  dm b = dm::rand(1);
  // Create symbolic constraint
  sym f = sym(c) * x + sym(b);

  // Create linear constraint function
  dopt::LinearCost::SharedPtr con =
      std::make_shared<dcas::LinearCost>("lc", f, x);

  // Create binding
  dsym::Variable z("z"), y("y");

  dsym::Vector v1(1), v2(1);
  v1 << z;
  v2 << y;

  Eigen::VectorXd in(1);
  in << 1.0;

  dopt::Binding<dopt::LinearCost> b1(con, v1), b2(con, v2);
  dopt::Binding<dopt::Cost> b3(b1), b4(b2);
}

TEST(Optimisation, CastBindingCost) {
  sym x = sym::sym("x", 1);
  dm c = dm::rand(1);
  dm b = dm::rand(1);
  // Create symbolic constraint
  sym f = sym(c) * x + sym(b);

  // Create linear constraint function
  dopt::LinearCost::SharedPtr con =
      std::make_shared<dcas::LinearCost>("lc", f, x);

  // Create binding
  dsym::Variable z("z");

  dsym::Vector v1(1);
  v1 << z;

  Eigen::VectorXd in(1);
  in << 1.0;

  dopt::Binding<dopt::LinearCost> linear_binding(con, v1);
  dopt::Binding<dopt::Cost> base_binding(linear_binding);

  // Evaluate as base class binding
  double res = base_binding.get()->evaluate(in);
  EXPECT_EQ(res, c->at(0) * in[0] + b->at(0));

  // Check coeffcients from original binding
  Eigen::VectorXd a1(1);
  double a2 = 0.0;
  linear_binding.get()->coeffs(a1, a2);

  EXPECT_EQ(a1[0], c->at(0));
  EXPECT_EQ(a2, b->at(0));
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  int status = RUN_ALL_TESTS();
  damotion::Profiler summary;
  return status;
}