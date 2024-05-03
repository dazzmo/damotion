#define DAMOTION_USE_PROFILING
#include "damotion/control/osc/osc.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "damotion/common/function.h"
#include "damotion/solvers/solver.h"
#include "damotion/utils/pinocchio_model.h"
#include "pinocchio/parsers/urdf.hpp"

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
  opt::QuadraticCost<Eigen::MatrixXd>::SharedPtr cost =
      std::make_shared<opt::QuadraticCost<Eigen::MatrixXd>>("task_cost", obj);

  // Create random configuration and velocity and desired task acceleration
  Eigen::VectorXd q = pinocchio::randomConfiguration(model);
  Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  Eigen::VectorXd a = Eigen::VectorXd::Random(model.nv);
  Eigen::VectorXd e = Eigen::VectorXd::Random(6);

  LOG(INFO) << "q: " << q.transpose() << std::endl;
  LOG(INFO) << "v: " << v.transpose() << std::endl;
  LOG(INFO) << "a: " << a.transpose() << std::endl;
  LOG(INFO) << "e: " << e.transpose() << std::endl;

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

  // Create input
  damotion::common::InputRefVector x = {a, q, v, e};
  cost->eval(x, {}, true);
  cost->eval_hessian(x, {});

  Eigen::MatrixXd A = J;
  Eigen::VectorXd b = dJdt_v;

  Eigen::MatrixXd Q_true = 2.0 * A.transpose() * A;
  Eigen::VectorXd g_true = 2.0 * b.transpose() * A;
  double c_true = b.dot(b);
  double cost_true = (A * a + b).squaredNorm();

  LOG(INFO) << "A:\n" << A << std::endl;
  LOG(INFO) << "b:\n" << b.transpose() << std::endl;

  LOG(INFO) << Q_true << std::endl;
  LOG(INFO) << cost->A() << std::endl;
  LOG(INFO) << g_true.transpose() << std::endl;
  LOG(INFO) << cost->b().transpose() << std::endl;
  LOG(INFO) << c_true << std::endl;
  LOG(INFO) << cost->c() << std::endl;

  EXPECT_TRUE(cost->A().isApprox(Q_true));
  EXPECT_TRUE(cost->b().isApprox(g_true));
  EXPECT_DOUBLE_EQ(cost->c(), c_true);
  EXPECT_DOUBLE_EQ(cost_true, cost->Objective());
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
