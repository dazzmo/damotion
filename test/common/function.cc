#define DAMOTION_USE_PROFILING

#include "damotion/common/function.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/common/logging.h"

namespace common = damotion::common;

class TestCallback {
 public:
  void foo(const common::std::vector<ConstVectorRef> &in,
           std::vector<Eigen::VectorXd> &out) {
    out[0] = in[0] + in[1];
  }

 private:
};

TEST(Function, CallbackFunction) {
  common::Function<Eigen::VectorXd>::UniquePtr f;

  // Create class
  TestCallback b;

  f = std::make_unique<common::CallbackFunction<Eigen::VectorXd>>(
      2, 1,
      [&b](const common::std::vector<ConstVectorRef> &in,
           std::vector<Eigen::VectorXd> &out) { b.foo(in, out); });

  // Create dummy input
  Eigen::VectorXd xa(5), xb(5), xc(5);
  xa.setRandom();
  xb.setRandom();

  LOG(INFO) << xa.transpose();
  LOG(INFO) << xb.transpose();

  xc = xa + xb;

  // Set function
  f->call({xa, xb});

  EXPECT_TRUE(xc.isApprox(f->getOutput(0)));
}

TEST(Function, NoCallbackFunction) {
  common::Function<Eigen::VectorXd>::UniquePtr f;

  // Create dummy input
  Eigen::VectorXd xa(5), xb(5), xc(5);
  xa.setRandom();
  xb.setRandom();

  xc = xa + xb;

  // Set function
  try {
    f->call({xa, xb});
  } catch (const std::runtime_error &error) {
    EXPECT_TRUE(error.what() ==
                "Function callback not provided to CallbackFunction!");
  }
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
