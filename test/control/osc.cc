#define DAMOTION_USE_PROFILING
#include "control/osc/osc.h"

#include <gtest/gtest.h>

#include "pinocchio/parsers/urdf.hpp"
#include "utils/pinocchio_model.h"

TEST(OSC, Parameters) {
    damotion::control::OSCController osc;

    osc.AddParameters("a", 5);
    osc.AddParameters("b", 3);
    osc.AddParameters("c", 2);
    osc.AddParameters("b", 7);
    osc.AddParameters("qacc", 9);

    osc.ListParameters();

    damotion::common::Profiler data;

    EXPECT_TRUE(true);
}

TEST(OSC, Cost) {
    casadi::SX qacc = casadi::SX::sym("qacc", 1);
    casadi::SX ctrl = casadi::SX::sym("ctrl", 1);
    casadi::SX lam = casadi::SX::sym("lam", 1);
    casadi::SX a = casadi::SX::sym("a", 1);
    casadi::SX b = casadi::SX::sym("b", 1);

    casadi::SX x = casadi::SX::vertcat({qacc, ctrl, lam});

    casadi::SX J = qacc * ctrl + (a + lam) * b;

    damotion::control::OSCController::Cost cost(
        "cost", J, {qacc, ctrl, lam, a, b}, {"qacc", "ctrl", "lam", "a", "b"},
        x);

    EXPECT_TRUE(true);
}

TEST(OSC, AddEndEffector) {
    // Load UR10 arm
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, true);
    pinocchio::Data data(model);

    casadi_utils::PinocchioModelWrapper wrapper(model);

    // Create OSC with system dimensions
    damotion::control::OSCController osc(model.nq, model.nv, model.nv);

    // Create data for end-effector
    wrapper.addEndEffector("tool0");

    // Create actuation map (use model if needed)
    casadi::SX B(6, 6);
    for (int i = 0; i < 6; i++) {
        B(i, i) = model.rotorGearRatio(i);
    }

    // Compute symbolic expression for system dynamics
    casadi::SX dyn = wrapper.rnea()(
        casadi::SXVector({osc.GetParameters("qpos"), osc.GetParameters("qvel"),
                          osc.GetVariables("qacc")}))[0];

    // Add any additional nonlinearities (e.g. spring/damping of joints)

    // Add generalised inputs
    dyn -= mtimes(B, osc.GetVariables("ctrl"));

    // Create new function with actuation included
    casadi::Function controlled_dynamics =
        casadi::Function("dynamics",
                         {osc.GetParameters("qpos"), osc.GetParameters("qvel"),
                          osc.GetVariables("qacc"), osc.GetVariables("ctrl")},
                         {dyn}, {"qpos", "qvel", "qacc", "ctrl"}, {"dyn"});

    osc.AddDynamics(controlled_dynamics);
    // Add tracking tasks
    osc.AddTrackingTask(
        "tool0", wrapper.end_effector(0).x,
        damotion::control::OSCController::TrackingTask::Type::kFull);

    // Add a contact task
    osc.AddContactTask("tool0", wrapper.end_effector(0).x);

    osc.Initialise();

    osc.ListParameters();
    osc.ListVariables();
    osc.ListCosts();
    osc.ListConstraints();

    // Set a desired pose
    osc.UpdateTrackingReference("tool0", Eigen::Vector3d(0, 1.0, 0.0),
                                damotion::control::RPYToQuaterion(0, 0, 0));
    Eigen::DiagonalMatrix<double, 6> Kp, Kd;
    Kp.diagonal().setConstant(1e1);
    Kd.diagonal().setConstant(1e0);
    osc.UpdateTrackingCostGains("tool0", Kp, Kd);

    osc.UpdateContactFrictionCoefficient("tool0", 1.0);

    // Generate program
    osc.UpdateProgramParameters();

    osc.ListParameters();

    osc.PrintProgramSummary();
    // Solve by passing the program to the solver

    EXPECT_TRUE(true);
}