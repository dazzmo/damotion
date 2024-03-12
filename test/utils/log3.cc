#include "utils/log3.h"

#include <gtest/gtest.h>

#include <pinocchio/spatial/se3.hpp>

#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"

TEST(Log3, CompareWithPinocchio) {
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
    casadi_utils::eigen::toCasadi(wR, res);

    // Generate function and codegen
    casadi::Function log3_map("log3", {q.w(), q.x(), q.y(), q.z()}, {res});
    log3_map = casadi_utils::codegen(log3_map, "./tmp");

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
    casadi_utils::eigen::toEigen(log3_map(in)[0], wd);

    // Compare values
    EXPECT_TRUE(wd.isApprox(wp));
}