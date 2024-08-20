#define DAMOTION_USE_PROFILING

#include "damotion/symbolic/variable.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/core/logging.hpp"
#include "damotion/core/profiler.hpp"

using sym = ::casadi::SX;
using dm = ::casadi::DM;

namespace ds = damotion::symbolic;

TEST(Symbolic, AddVariable) {
  ds::VariableVector vv;

  ds::Variable x("x"), y("y");

  EXPECT_TRUE(vv.add(x));
  EXPECT_TRUE(vv.add(y));
}

TEST(Symbolic, AddSameVariable) {
  ds::VariableVector vv;

  ds::Variable x("x"), y("y");

  vv.add(x);
  vv.add(y);
  EXPECT_FALSE(vv.add(x));
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  int status = RUN_ALL_TESTS();
  damotion::Profiler summary;
  return status;
}