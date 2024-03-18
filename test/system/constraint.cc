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

    // Get function for pose error
    // TODO: Look at how parameters should be passed
    casadi::SX q = casadi::SX::sym("qpos", wrapper.model().nq),
               v = casadi::SX::sym("qvel", wrapper.model().nv),
               a = casadi::SX::sym("qvel", wrapper.model().nv);

    casadi::SX qR = casadi::SX::sym("qref", 4), pR = casadi::SX::sym("pref", 3);

    // Compute functions for end-effector velocity and acceleration
    casadi::SX c = wrapper.end_effector(0).pose_error(
                   casadi::SXVector({q, qR, pR}))[0],
               dc = wrapper.end_effector(0).x(casadi::SXVector({q, v, a}))[1],
               ddc = wrapper.end_effector(0).x(casadi::SXVector({q, v, a}))[2],
               J = wrapper.end_effector(0).J(casadi::SXVector({q}))[0];

    // Create holonomic constraint
 
    EXPECT_TRUE(true);
}