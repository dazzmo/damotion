#define DAMOTION_USE_PROFILING

#include "damotion/core/function.hpp"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/core/logging.hpp"

using namespace damotion;

class NormFunction : public damotion::FunctionBase<1, double> {
 public:
  NormFunction() = default;

  double evaluate(const InputVectorType &x,
                  OptionalJacobianType J = nullptr) const {
    if (J) *J << 2.0 * x.transpose();
    return x.norm();
  }
};

TEST(Function, OptionalMatrixTests) {
  Eigen::MatrixXd A(2, 2);
  A.setRandom();

  OptionalMatrix oA(A);

  Eigen::RowVector2d b(2);
  OptionalRowVector ob(b);

  EXPECT_TRUE(A.isApprox(*oA));
  EXPECT_TRUE(b.isApprox(*ob));
}

TEST(Function, NormFunctionEvaluate) {
  NormFunction nf;

  // Create dummy input
  Eigen::VectorXd xa(3), xb(5), xc(20);
  xa.setRandom();
  xb.setRandom();
  xc.setRandom();

  EXPECT_TRUE(xa.norm() == nf.evaluate(xa));
  EXPECT_TRUE(xb.norm() == nf.evaluate(xb));
  EXPECT_TRUE(xc.norm() == nf.evaluate(xc));
}

TEST(Function, NormFunctionEvaluateJacobian) {
  NormFunction nf;

  // Create dummy input
  Eigen::VectorXd xa(3), xb(5), xc(20);
  Eigen::RowVectorXd Ja(1, 3), Jb(1, 5), Jc(1, 20);
  xa.setRandom();
  xb.setRandom();
  xc.setRandom();

  nf.evaluate(xa, Ja);
  nf.evaluate(xb, Jb);
  nf.evaluate(xc, Jc);

  EXPECT_TRUE(xa.norm() == nf.evaluate(xa, Ja));
  EXPECT_TRUE(xb.norm() == nf.evaluate(xb, Jb));
  EXPECT_TRUE(xc.norm() == nf.evaluate(xc, Jc));

  VLOG(10) << "Ja = " << Ja;
  VLOG(10) << "Jb = " << Jb;
  VLOG(10) << "Jc = " << Jc;
}

class VectorFunction : public damotion::FunctionBase<1, Eigen::VectorXd> {
 public:
  VectorFunction() = default;
  Eigen::VectorXd evaluate(const InputVectorType &x,
                           OptionalJacobianType J = nullptr) const {
    if (J) {
      // Compute Jacobian
    }
    return -x;
  }
};

TEST(Function, VectorFunctionEvaluate) {
  VectorFunction nf;

  // Create dummy input
  Eigen::VectorXd xa(3), xb(5), xc(20);
  xa.setRandom();
  xb.setRandom();
  xc.setRandom();

  EXPECT_TRUE(xa.isApprox(-nf.evaluate(xa)));
  EXPECT_TRUE(xb.isApprox(-nf.evaluate(xb)));
  EXPECT_TRUE(xc.isApprox(-nf.evaluate(xc)));
}

class BiVectorFunction : public damotion::FunctionBase<2, Eigen::VectorXd> {
 public:
  BiVectorFunction() = default;
  Eigen::VectorXd evaluate(const InputVectorType &x, const InputVectorType &y,
                           OptionalJacobianType Jx = nullptr,
                           OptionalJacobianType Jy = nullptr) const {
    if (Jx) {
      // Compute Jx
    }
    if (Jy) {
      // Compute Jy
    }
    return x + y;
  }
};

void foo(const Eigen::VectorXd &in) {
  const_cast<Eigen::VectorXd &>(in) << 1.0, 1.0, 1.0;
}

TEST(Function, BiVectorFunctionEvaluate) {
  BiVectorFunction nf;

  // Create dummy input
  Eigen::VectorXd xa(3), xb(3), xc(3);
  xa.setRandom();
  xb.setRandom();
  xc.setRandom();

  EXPECT_TRUE((xa + xb).isApprox(nf.evaluate(xa, xb)));
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
