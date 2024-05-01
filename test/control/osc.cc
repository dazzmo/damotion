#define DAMOTION_USE_PROFILING
#include "control/osc/osc.h"

#include <gtest/gtest.h>

#include "pinocchio/parsers/urdf.hpp"
#include "solvers/solver.h"
#include "utils/pinocchio_model.h"

TEST(TrackingCost, QuadraticForm) {
  pinocchio::Model model;
  pinocchio::urdf::buildModel("./ur10_robot.urdf", model, false);
  pinocchio::Data data(model);

  damotion::utils::casadi::PinocchioModelWrapper wrapper(model);

  // Create end-effector
  auto tool0 = wrapper.AddEndEffector("tool0");

  // Create expression of the form || J qacc + dJdt * qvel - xaccd ||^2
  casadi::SX qpos = casadi::SX::sym("qpos", model.nq),
             qvel = casadi::SX::sym("qvel", model.nv),
             qacc = casadi::SX::sym("qacc", model.nv);

  casadi::SX xaccd = casadi::SX::sym("xaccd", 6);
  tool0->UpdateState(qpos, qvel, qacc);

  sym::Expression obj = mtimes(tool0->acc_sym().T(), tool0->acc_sym());
  obj.SetInputs({qacc}, {qpos, qvel, xaccd});

  // Create quadratic cost
  opt::QuadraticCost cost("task_cost", obj);

  // Create random configuration and velocity and desired task acceleration
  Eigen::VectorXd q = pinocchio::randomConfiguration(model);
  Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  Eigen::VectorXd a = Eigen::VectorXd::Random(model.nv);
  Eigen::VectorXd e = Eigen::VectorXd::Random(6);

  std::cout << "q: " << q.transpose() << std::endl;
  std::cout << "v: " << v.transpose() << std::endl;
  std::cout << "a: " << a.transpose() << std::endl;
  std::cout << "e: " << e.transpose() << std::endl;

  // Evaluate the true system
  Eigen::MatrixXd J(6, model.nv);
  J.setZero();
  pinocchio::computeFrameJacobian(model, data, q, model.getFrameId("tool0"),
                                  pinocchio::LOCAL_WORLD_ALIGNED, J);
  // Get classical frame acceleration drift
  pinocchio::forwardKinematics(model, data, q, v,
                               Eigen::VectorXd::Zero(model.nv));
  Eigen::VectorXd dJdt_v = pinocchio::getFrameClassicalAcceleration(
                               model, data, model.getFrameId("tool0"),
                               pinocchio::LOCAL_WORLD_ALIGNED)
                               .toVector();

  cost.HessianFunction().setInput(0, a.data());
  cost.HessianFunction().setInput(1, q.data());
  cost.HessianFunction().setInput(2, v.data());
  cost.HessianFunction().setInput(3, e.data());
  cost.HessianFunction().call();

  cost.GradientFunction().setInput(0, a.data());
  cost.GradientFunction().setInput(1, q.data());
  cost.GradientFunction().setInput(2, v.data());
  cost.GradientFunction().setInput(3, e.data());
  cost.GradientFunction().call();

  cost.ObjectiveFunction().setInput(0, a.data());
  cost.ObjectiveFunction().setInput(1, q.data());
  cost.ObjectiveFunction().setInput(2, v.data());
  cost.ObjectiveFunction().setInput(3, e.data());
  cost.ObjectiveFunction().call();

  Eigen::MatrixXd A = J;
  Eigen::VectorXd b = dJdt_v;

  Eigen::MatrixXd Q_true = 2.0 * A.transpose() * A;
  Eigen::VectorXd g_true = 2.0 * b.transpose() * A;
  double c_true = b.dot(b);
  double cost_true = (A * a + b).squaredNorm();

  std::cout << "A:\n" << A << std::endl;
  std::cout << "b:\n" << b.transpose() << std::endl;

  std::cout << Q_true << std::endl;
  std::cout << cost.Q() << std::endl;
  std::cout << g_true.transpose() << std::endl;
  std::cout << cost.g().transpose() << std::endl;
  std::cout << c_true << std::endl;
  std::cout << cost.c() << std::endl;

  EXPECT_TRUE(cost.Q().isApprox(Q_true));
  EXPECT_TRUE(cost.g().isApprox(g_true));
  EXPECT_DOUBLE_EQ(cost.c(), c_true);
  EXPECT_DOUBLE_EQ(cost_true, cost.ObjectiveFunction().getOutput(0).data()[0]);
}
