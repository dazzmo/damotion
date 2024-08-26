
#include "damotion/casadi/eigen.hpp"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/core/logging.hpp"
#include "damotion/core/profiler.hpp"

using SX = ::casadi::SX;
using DM = ::casadi::DM;

TEST(CasadiEigen, EigenToCasadiDM) {
  std::size_t iterations = 1000;
  std::size_t rows = 10, cols = 5;

  Eigen::MatrixXd M(rows, cols);
  DM Mc;

  M.setRandom();

  for (std::size_t i = 0; i < iterations; ++i) {
    damotion::Profiler profiler("EigenToCasadiDM");
    damotion::casadi::toCasadi(M, Mc);
  }

  std::cout << M << '\n';
  std::cout << Mc << '\n';

  // Compare results
  bool success = true;
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      const double &a = Mc(i, j)->at(0);
      const double &b = M(i, j);

      if (std::abs(a - b) > std::numeric_limits<double>::epsilon()) {
        success = false;
      }
    }
  }

  EXPECT_TRUE(success);
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  int status = RUN_ALL_TESTS();
  damotion::Profiler summary;
  return status;
}
