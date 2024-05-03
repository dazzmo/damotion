#include "damotion/system/system.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

TEST(System, DerivedClass) { EXPECT_TRUE(false); }

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
