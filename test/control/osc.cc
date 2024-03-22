#define DAMOTION_USE_PROFILING
#include "control/osc/osc.h"

#include <gtest/gtest.h>

#include "pinocchio/parsers/urdf.hpp"
#include "solvers/solver.h"
#include "utils/pinocchio_model.h"

class OSCTest : public testing::Test {
   protected:
    void SetUp() override {
        damotion::control::OSCController osc0_;

        // Load UR10 arm
        pinocchio::urdf::buildModel("./ur10_robot.urdf", model_, true);
        data_ = pinocchio::Data(model_);
        // Wrap model
        wrapper_ = utils::casadi::PinocchioModelWrapper(model_);
        // Create OSC
        damotion::control::OSCController osc_(model_.nq, model_.nv, model_.nv);
    }

    // void TearDown() override {}

    damotion::control::OSCController osc0_;
    damotion::control::OSCController osc_;

    utils::casadi::PinocchioModelWrapper wrapper_;

    pinocchio::Model model_;
    pinocchio::Data data_;
};

TEST_F(OSCTest, IsEmptyInitially) {
    EXPECT_EQ(osc0_.NumberOfDecisionVariables(), 0);
    EXPECT_EQ(osc0_.NumberOfConstraints(), 0);
}

TEST_F(OSCTest, CorrectModelDimensions) {
    EXPECT_EQ(osc_.nq(), model_.nq);
    EXPECT_EQ(osc_.nv(), model_.nv);
}

TEST_F(OSCTest, AddDynamics) {
    // Create actuation map (use model if needed)
    casadi::SX B(6, 6);
    for (int i = 0; i < 6; i++) {
        B(i, i) = model_.rotorGearRatio(i);
    }

    // Compute symbolic expression for system dynamics
    casadi::SX dyn = wrapper_.rnea()(casadi::SXVector(
        {osc_.GetParameters("qpos"), osc_.GetParameters("qvel"),
         osc_.GetVariables("qacc")}))[0];

    // Add any additional nonlinearities (e.g. spring/damping of joints)

    // Add generalised inputs
    dyn -= mtimes(B, osc_.GetVariables("ctrl"));

    // Create new function with actuation included
    casadi::Function controlled_dynamics = casadi::Function(
        "dynamics",
        {osc_.GetParameters("qpos"), osc_.GetParameters("qvel"),
         osc_.GetVariables("qacc"), osc_.GetVariables("ctrl")},
        {dyn}, {"qpos", "qvel", "qacc", "ctrl"}, {"dyn"});

    osc_.AddDynamics(controlled_dynamics);

    EXPECT_TRUE(true);
}

TEST_F(OSCTest, AddTrackingTask) {
    // Create data for end-effector
    wrapper_.addEndEffector("tool0");

    // Add tracking tasks
    osc_.AddTrackingTask(
        "tool0", wrapper_.end_effector(0).x,
        damotion::control::OSCController::TrackingTask::Type::kFull);

    EXPECT_TRUE(true);
}

/*
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

    damotion::solvers::SolverBase solver(osc);

    // Generate program
    osc.UpdateProgramParameters();
    // Update solver
    solver.UpdateProgram(osc);

    osc.ListParameters();
    osc.PrintProgramSummary();


*/