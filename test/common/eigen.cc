#define DAMOTION_USE_PROFILING

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <Eigen/Core>

#include "damotion/common/logging.h"

TEST(Function, CallbackFunction) {
  Eigen::VectorXd vec = Eigen::VectorXd::Random(10);
  std::vector<int> indices;

  auto refVec = vec(indices);

  LOG(INFO) << vec.transpose();
  LOG(INFO) << refVec.transpose();
  refVec[0] += 1.0;
  LOG(INFO) << vec.transpose();
  LOG(INFO) << refVec.transpose();
  EXPECT_TRUE(true);
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
