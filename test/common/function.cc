#define DAMOTION_USE_PROFILING

#include "common/function.h"

#include <gtest/gtest.h>

#include "common/logging.h"

namespace common = damotion::common;

class TestCallback {
   public:
    void foo(const common::Function::InputRefVector &in,
             std::vector<Eigen::MatrixXd> &out) {
        out[0] = in[0] + in[1];
    }

   private:
};

TEST(Function, CallbackFunction) {
    std::unique_ptr<common::Function> f;

    // Create class
    TestCallback b;

    f = std::make_unique<common::CallbackFunction>(
        2, 1,
        [&b](const common::Function::InputRefVector &in,
             std::vector<Eigen::MatrixXd> &out) { b.foo(in, out); });

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
    std::unique_ptr<common::Function> f;

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