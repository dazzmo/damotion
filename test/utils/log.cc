#include "damotion/utils/log.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <pinocchio/spatial/se3.hpp>

#include "damotion/utils/codegen.h"
#include "damotion/utils/eigen_wrapper.h"

TEST(Log, Log3) {
  // Create codegen function
  // Create a rotation matrix
  Eigen::Quaternion<casadi::SX> q;
  q.w() = casadi::SX::sym("w");
  q.x() = casadi::SX::sym("x");
  q.y() = casadi::SX::sym("y");
  q.z() = casadi::SX::sym("z");
  Eigen::Matrix<casadi::SX, 3, 3> R = q.toRotationMatrix();
  Eigen::Vector<casadi::SX, 3> wR = damotion::log3(R);

  // Create function
  casadi::SX res;
  damotion::utils::casadi::toCasadi(wR, res);

  // Generate function and codegen
  casadi::Function flog3("log3", {q.w(), q.x(), q.y(), q.z()}, {res});
  flog3 = damotion::utils::casadi::codegen(flog3, "./tmp");

  // Generate random SE3 configuration and extract rotation matrix
  pinocchio::SE3 se3;
  se3.setRandom();
  Eigen::Matrix3d se3R = se3.rotation();

  // Compute quaternion of rotation
  Eigen::Quaterniond qR(se3R);

  // Compute log of rotation both ways and compare
  // Pinocchio
  Eigen::Vector3d wp = pinocchio::log3(se3R);
  // Damotion
  Eigen::VectorXd wd;
  casadi::DMVector in = {qR.w(), qR.x(), qR.y(), qR.z()};
  damotion::utils::casadi::toEigen(flog3(in)[0], wd);

  // Compare values
  EXPECT_TRUE(wd.isApprox(wp));
}

TEST(Log, JLog3) {
  // Create codegen function
  // Create a rotation matrix
  Eigen::Quaternion<casadi::SX> q;
  q.w() = casadi::SX::sym("q.w");
  q.x() = casadi::SX::sym("q.x");
  q.y() = casadi::SX::sym("q.y");
  q.z() = casadi::SX::sym("q.z");

  Eigen::Matrix<casadi::SX, 3, 3> R = q.toRotationMatrix();
  Eigen::Matrix<casadi::SX, 3, 3> Jlog;
  casadi::SX theta;
  Eigen::Matrix<casadi::SX, 3, 1> log = damotion::log3(R, theta);
  damotion::Jlog3(theta, log, Jlog);

  // Create function
  casadi::SX res;
  damotion::utils::casadi::toCasadi(Jlog, res);

  // Generate function and codegen
  casadi::Function flog3("Jlog3", {q.w(), q.x(), q.y(), q.z()}, {res});
  flog3 = damotion::utils::casadi::codegen(flog3, "./tmp");

  // Generate random SE3 configuration and extract rotation matrix
  pinocchio::SE3 se3;
  se3.setRandom();
  Eigen::Matrix3d se3R = se3.rotation();
  // Compute quaternion of rotation
  Eigen::Quaterniond qR(se3R);

  // Compute log of rotation both ways and compare
  // Pinocchio
  Eigen::Matrix<double, 3, 3> wp;
  double theta_d = 0;
  Eigen::Vector3d log_d = pinocchio::log3(se3R, theta_d);
  pinocchio::Jlog3(theta_d, log_d, wp);
  // Damotion
  Eigen::MatrixXd wd;
  casadi::DMVector in = {qR.w(), qR.x(), qR.y(), qR.z()};
  damotion::utils::casadi::toEigen(flog3(in)[0], wd);

  // Compare values
  EXPECT_TRUE(wd.isApprox(wp));
}

TEST(Log, Log6) {
  // Create codegen function
  // Create a rotation matrix
  Eigen::Quaternion<casadi::SX> q;
  q.w() = casadi::SX::sym("q.w");
  q.x() = casadi::SX::sym("q.x");
  q.y() = casadi::SX::sym("q.y");
  q.z() = casadi::SX::sym("q.z");
  Eigen::Vector3<casadi::SX> x;
  x.x() = casadi::SX::sym("x");
  x.y() = casadi::SX::sym("y");
  x.z() = casadi::SX::sym("z");

  Eigen::Matrix<casadi::SX, 3, 3> R = q.toRotationMatrix();
  Eigen::Vector<casadi::SX, 6> log = damotion::log6(R, x);

  // Create function
  casadi::SX res;
  damotion::utils::casadi::toCasadi(log, res);

  // Generate function and codegen
  casadi::Function flog6(
      "log6", {x.x(), x.y(), x.z(), q.w(), q.x(), q.y(), q.z()}, {res});
  flog6 = damotion::utils::casadi::codegen(flog6, "./tmp");

  // Generate random SE3 configuration and extract rotation matrix
  pinocchio::SE3 se3;
  se3.setRandom();
  Eigen::Matrix3d se3R = se3.rotation();
  // Compute quaternion of rotation
  Eigen::Quaterniond qR(se3R);

  // Compute log of rotation both ways and compare
  // Pinocchio
  Eigen::Vector<double, 6> wp = pinocchio::log6(se3).toVector();
  // Damotion
  Eigen::VectorXd wd;
  casadi::DMVector in = {se3.translation().x(),
                         se3.translation().y(),
                         se3.translation().z(),
                         qR.w(),
                         qR.x(),
                         qR.y(),
                         qR.z()};
  damotion::utils::casadi::toEigen(flog6(in)[0], wd);

  // Compare values
  EXPECT_TRUE(wd.isApprox(wp));
}

TEST(Log, JLog6) {
  // Create codegen function
  // Create a rotation matrix
  Eigen::Quaternion<casadi::SX> q;
  q.w() = casadi::SX::sym("q.w");
  q.x() = casadi::SX::sym("q.x");
  q.y() = casadi::SX::sym("q.y");
  q.z() = casadi::SX::sym("q.z");
  Eigen::Vector3<casadi::SX> x;
  x.x() = casadi::SX::sym("x");
  x.y() = casadi::SX::sym("y");
  x.z() = casadi::SX::sym("z");

  Eigen::Matrix<casadi::SX, 3, 3> R = q.toRotationMatrix();
  Eigen::Matrix<casadi::SX, 6, 6> Jlog;
  damotion::Jlog6(R, x, Jlog);

  std::cout << Jlog.bottomLeftCorner(3, 3) << std::endl;

  // Create function
  casadi::SX res;
  damotion::utils::casadi::toCasadi(Jlog, res);

  std::cout << res(casadi::Slice(3, 6), casadi::Slice(0, 3)) << std::endl;

  // Generate function and codegen
  casadi::Function log6_map(
      "Jlog6", {x.x(), x.y(), x.z(), q.w(), q.x(), q.y(), q.z()}, {res});
  log6_map = damotion::utils::casadi::codegen(log6_map, "./tmp");

  // Generate random SE3 configuration and extract rotation matrix
  pinocchio::SE3 se3;
  se3.setRandom();
  Eigen::Matrix3d se3R = se3.rotation();
  // Compute quaternion of rotation
  Eigen::Quaterniond qR(se3R);

  // Compute log of rotation both ways and compare
  // Pinocchio
  Eigen::Matrix<double, 6, 6> wp;
  pinocchio::Jlog6(se3, wp);
  // Damotion
  Eigen::MatrixXd wd;
  casadi::DMVector in = {se3.translation().x(),
                         se3.translation().y(),
                         se3.translation().z(),
                         qR.w(),
                         qR.x(),
                         qR.y(),
                         qR.z()};

  std::cout << log6_map(in)[0](casadi::Slice(3, 6), casadi::Slice(0, 3))
            << std::endl;
  damotion::utils::casadi::toEigen(log6_map(in)[0], wd);

  std::cout << wp << std::endl;
  std::cout << wd << std::endl;

  // Compare values
  EXPECT_TRUE(wd.isApprox(wp));
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
