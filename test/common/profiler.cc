#include "common/profiler.h"
#include <boost/mpl/accumulate.hpp>

#include <gtest/gtest.h>

TEST(Profiler, Basic) {
    for (int i = 0; i < 100; ++i) {
        damotion::common::Profiler profile1("loop1");

        for (int j = 0; j < 50; ++j) {
            damotion::common::Profiler profile1("loop2");
        }
    }

    damotion::common::Profiler profiler_summary;

    EXPECT_TRUE(true);
}
