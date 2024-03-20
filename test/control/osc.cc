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

    casadi::Function f("f", casadi::SXVector({qacc, ctrl, lam, a, b}),
                       casadi::SXVector({J}), {"qacc", "ctrl", "lam", "a", "b"},
                       {"c"});

    damotion::control::OSCController::Cost cost(f.name(), f, x);

    EXPECT_TRUE(true);
}

TEST(OSC, AddEndEffector) {
    // Load UR10 arm
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, true);
    pinocchio::Data data(model);

    casadi_utils::PinocchioModelWrapper wrapper(model);

    // Create actuation map
    casadi::SX B(6, 6);
    for (int i = 0; i < 6; i++) {
        B(i, i) = 1.0;
    }

    casadi::SX qpos = casadi::SX::sym("qpos", model.nq),
               qvel = casadi::SX::sym("qvel", model.nv),
               qacc = casadi::SX::sym("qacc", model.nv),
               u = casadi::SX::sym("ctrl", model.nv);

    casadi::SX dyn = wrapper.rnea()(casadi::SXVector({qpos, qvel, qacc}))[0];
    dyn -= mtimes(B, u);
    // Create new function with actuation included
    casadi::Function controlled_dynamics =
        casadi::Function("dynamics", {qpos, qvel, qacc, u}, {dyn},
                         {"qpos", "qvel", "qacc", "ctrl"}, {"dyn"});

    // Create data for end-effector
    wrapper.addEndEffector("tool0");

    damotion::control::OSCController osc(model.nq, model.nv, model.nv);

    osc.AddDynamics(controlled_dynamics);
    osc.AddTrackingTask(
        "tool0", wrapper.end_effector(0).x,
        damotion::control::OSCController::TrackingTask::Type::kFull);

    // Observe parameters and cost
    osc.ListVariables();
    osc.ListParameters();

    osc.Initialise();

    EXPECT_TRUE(true);
}