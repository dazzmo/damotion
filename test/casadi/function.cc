#define DAMOTION_USE_PROFILING

#include "damotion/casadi/function.hpp"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/core/logging.hpp"

using sym = ::casadi::SX;

TEST(CasadiFunction, NormFunctionEvaluate) {
  sym x = sym::sym("x", 2);
  sym c = 2.0 * x + 1.0;

  // Create linear constraint function
  damotion::optimisation::LinearConstraint::SharedPtr p =
      std::make_shared<damotion::casadi::LinearConstraint>(c, x);
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
