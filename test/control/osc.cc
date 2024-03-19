#define DAMOTION_USE_PROFILING
#include "control/osc/osc.h"

#include <gtest/gtest.h>

#include "pinocchio/parsers/urdf.hpp"
#include "utils/pinocchio_model.h"

TEST(OSC, Parameters) {
    OSCController osc;

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

    OSCController::Cost cost(f, x);

    std::cout << cost.g.f()(casadi::SXVector({qacc, ctrl, lam, a, b}))
              << std::endl;
    std::cout << cost.H.f()(casadi::SXVector({qacc, ctrl, lam, a, b}))
              << std::endl;

    EXPECT_TRUE(true);
}

TEST(OSC, AddEndEffector) {
    // Load UR10 arm
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, true);
    pinocchio::Data data(model);

    casadi_utils::PinocchioModelWrapper wrapper(model);

    // Create data for end-effector
    wrapper.addEndEffector("tool0");

    OSCController osc(model.nq, model.nv, model.nv);

    // Add system dynamics
    casadi::Function f = wrapper.rnea();
    
    osc.AddDynamics(f);
    osc.AddTrackingTask("tool0", wrapper.end_effector(0).x,
                        OSCController::TrackingTask::Type::kFull);

    osc.Initialise();

    // Observe parameters and cost
    osc.ListVariables();
    osc.ListParameters();

    EXPECT_TRUE(true);
}