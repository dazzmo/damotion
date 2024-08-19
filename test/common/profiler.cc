#define DAMOTION_USE_PROFILING

#include "damotion/core/profiler.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <boost/mpl/accumulate.hpp>

#include "damotion/core/logging.h"

TEST(Profiler, Basic) {
  for (int i = 0; i < 100; ++i) {
    damotion::core::Profiler profile1("loop1");

    for (int j = 0; j < 50; ++j) {
      damotion::core::Profiler profile1("loop2");
      damotion::core::Profiler profile2("loop3");
    }
  }

  damotion::core::Profiler profiler_summary;

  EXPECT_TRUE(true);
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
