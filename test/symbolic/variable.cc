#define DAMOTION_USE_PROFILING

#include "damotion/symbolic/variable.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/core/logging.hpp"
#include "damotion/core/profiler.hpp"

using sym = ::casadi::SX;
using dm = ::casadi::DM;

namespace ds = damotion::symbolic;

TEST(VariableVector, AddVariable) {
  ds::VariableVector vv;

  ds::Variable x("x"), y("y");

  EXPECT_TRUE(vv.add(x));
  EXPECT_TRUE(vv.add(y));

  EXPECT_EQ(vv.getIndex(x), 0);
  EXPECT_EQ(vv.getIndex(y), 1);

}

TEST(VariableVector, AddSameVariable) {
  ds::VariableVector vv;

  ds::Variable x("x"), y("y");

  vv.add(x);
  vv.add(y);
  EXPECT_FALSE(vv.add(x));
}

TEST(VariableVector, AddAndRemoveVariable) {
  ds::VariableVector vv;

  ds::Variable x("x"), y("y");

  vv.add(x);
  vv.add(y);

  EXPECT_EQ(vv.size(), 2);
  EXPECT_TRUE(vv.contains(x));
  EXPECT_TRUE(vv.contains(y));

  EXPECT_TRUE(vv.remove(x));

  EXPECT_EQ(vv.size(), 1);
  EXPECT_FALSE(vv.contains(x));
}

TEST(VariableVector, AddMultipleVariables) {
  ds::VariableVector vv;

  ds::Vector x = ds::createVector("x", 5);

  vv.add(x);

  EXPECT_EQ(vv.size(), 5);
  EXPECT_TRUE(vv.contains(x[0]));
  EXPECT_TRUE(vv.contains(x[4]));
}

TEST(VariableVector, AddAndRemoveMultipleVariables) {
  ds::VariableVector vv;

  ds::Vector x = ds::createVector("x", 5);

  vv.add(x);

  EXPECT_EQ(vv.size(), 5);
  EXPECT_TRUE(vv.contains(x[0]));
  EXPECT_TRUE(vv.contains(x[4]));

  vv.remove(x.bottomRows(2));

  EXPECT_EQ(vv.size(), 3);
  EXPECT_TRUE(vv.contains(x[0]));
  EXPECT_FALSE(vv.contains(x[4]));
}

TEST(VariableVector, CustomReordering) {
  ds::VariableVector vv;

  ds::Variable a("a"), b("b"), c("c"), d("d"), e("e");

  vv.add(a);
  vv.add(b);
  vv.add(c);
  vv.add(d);
  vv.add(e);

  ds::Vector reordering(5);
  reordering << b, a, d, e, c;
  vv.reorder(reordering);

  // EXPECT_EQ(vv.getIndex(a), 1);
  EXPECT_EQ(vv.getIndex(b), 0);
  // EXPECT_EQ(vv.getIndex(c), 4);
  // EXPECT_EQ(vv.getIndex(d), 2);
  // EXPECT_EQ(vv.getIndex(e), 3);
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  int status = RUN_ALL_TESTS();
  damotion::Profiler summary;
  return status;
}