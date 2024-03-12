#include "system/constraint.h"

#include <gtest/gtest.h>

#include <pinocchio/parsers/urdf.hpp>

#include "utils/pinocchio_model.h"

TEST(Constraint, ConstraintCreation) {
    std::string name = "constraint";
    int sz = 3;
    damotion::system::Constraint c(name, sz);

    EXPECT_TRUE(c.name() == name);
    EXPECT_TRUE(c.nc() == sz);
}

TEST(HolonomicConstraint, ConstraintCreation) {
    std::string name = "constraint";
    int sz = 3;
    int nq = 5;
    int nv = 4;
    damotion::system::HolonomicConstraint c(name, sz, nq, nv);

    EXPECT_TRUE(c.name() == name);
    EXPECT_TRUE(c.nc() == sz);
    EXPECT_TRUE(c.nq() == nq);
    EXPECT_TRUE(c.nv() == nv);
}

TEST(HolonomicConstraint, ConstraintCreationEndEffector) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, true);
    pinocchio::Data data(model);

    casadi_utils::PinocchioModelWrapper wrapper(model);

    // Add an end-effector
    wrapper.addEndEffector("tool0");

    std::cout << "End effector added\n";
    Eigen::Matrix<double, 6, 3> S;
    S.setZero();
    S.topRows(3).setIdentity();
    // Create holonomic constraint for 3D position
    wrapper.end_effector(wrapper.end_effector_idx("tool0")).S = S;

    std::cout << "Making constraint\n";
    damotion::system::HolonomicConstraint c(
        model.name + "tool0_constraint",
        wrapper.end_effector(wrapper.end_effector_idx("tool0")));

    // Create target pose
    Eigen::Vector<casadi::SX, 3> tR;
    tR.x() = casadi::SX::sym("x.x");
    tR.y() = casadi::SX::sym("x.y");
    tR.z() = casadi::SX::sym("x.z");
    Eigen::Quaternion<casadi::SX> qR;
    qR.w() = casadi::SX::sym("q.w");
    qR.x() = casadi::SX::sym("q.x");
    qR.y() = casadi::SX::sym("q.y");
    qR.z() = casadi::SX::sym("q.z");
    // Create SE3
    pinocchio::SE3Tpl<casadi::SX> Tr(qR, tR);
    pinocchio::SE3Tpl<casadi::SX> err = Tr.actInv(wrapper.end_effector(0).pose);
    Eigen::Vector<casadi::SX, 6> err_se3 = pinocchio::log6(err.toHomogeneousMatrix());

    std::cout << "Printing\n";
    std::cout << c.constraint() << std::endl;
    std::cout << c.firstTimeDerivative() << std::endl;
    std::cout << c.secondTimeDerivative() << std::endl;
    std::cout << c.jacobian() << std::endl;

    EXPECT_TRUE(true);
}