#include "utils/pinocchio_model.h"

#include <gtest/gtest.h>

#include "pinocchio/parsers/urdf.hpp"
#include "utils/codegen.h"


TEST(PinocchioModelWrapper, LoadModel) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, true);
    pinocchio::Data data(model);
    EXPECT_TRUE(true);
}

TEST(PinocchioModelWrapper, ABATest) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, false);
    pinocchio::Data data(model);

    damotion::utils::casadi::PinocchioModelWrapper wrapper(model);

    casadi::Function aba = wrapper.aba();

    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v(model.nv);
    v.setRandom();
    Eigen::VectorXd tau(model.nv);
    tau.setRandom();

    Eigen::VectorXd a = pinocchio::aba(model, data, q, v, tau);

    // Compute through function
    casadi::DM qd, vd, ud;
    damotion::utils::casadi::toCasadi(q, qd);
    damotion::utils::casadi::toCasadi(v, vd);
    damotion::utils::casadi::toCasadi(tau, ud);

    casadi::DM ad = aba(casadi::DMVector({qd, vd, ud}))[0];

    Eigen::VectorXd ac;
    damotion::utils::casadi::toEigen(ad, ac);

    EXPECT_TRUE(a.isApprox(ac));
}

TEST(PinocchioModelWrapper, ABATestCodegen) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, false);
    pinocchio::Data data(model);

    damotion::utils::casadi::PinocchioModelWrapper wrapper(model);

    casadi::Function aba = wrapper.aba();
    aba = damotion::utils::casadi::codegen(aba, "./tmp/");

    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v(model.nv);
    v.setRandom();
    Eigen::VectorXd tau(model.nv);
    tau.setRandom();

    Eigen::VectorXd a = pinocchio::aba(model, data, q, v, tau);

    // Compute through function
    casadi::DM qd, vd, ud;
    damotion::utils::casadi::toCasadi(q, qd);
    damotion::utils::casadi::toCasadi(v, vd);
    damotion::utils::casadi::toCasadi(tau, ud);

    casadi::DM ad = aba(casadi::DMVector({qd, vd, ud}))[0];

    Eigen::VectorXd ac;
    damotion::utils::casadi::toEigen(ad, ac);

    EXPECT_TRUE(a.isApprox(ac));
}

TEST(PinocchioModelWrapper, RNEATest) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, false);
    pinocchio::Data data(model);

    damotion::utils::casadi::PinocchioModelWrapper wrapper(model);

    casadi::Function rnea = wrapper.rnea();

    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v(model.nv);
    v.setRandom();
    Eigen::VectorXd a(model.nv);
    a.setRandom();

    Eigen::VectorXd u = pinocchio::rnea(model, data, q, v, a);

    // Compute through function
    casadi::DM qd, vd, ad;
    damotion::utils::casadi::toCasadi(q, qd);
    damotion::utils::casadi::toCasadi(v, vd);
    damotion::utils::casadi::toCasadi(a, ad);

    casadi::DM ud = rnea(casadi::DMVector({qd, vd, ad}))[0];

    Eigen::VectorXd uc;
    damotion::utils::casadi::toEigen(ud, uc);

    EXPECT_TRUE(u.isApprox(uc));
}

TEST(PinocchioModelWrapper, RNEATestCodegen) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, false);
    pinocchio::Data data(model);

    damotion::utils::casadi::PinocchioModelWrapper wrapper(model);

    casadi::Function rnea = wrapper.rnea();
    rnea = damotion::utils::casadi::codegen(rnea, "./tmp/");

    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v(model.nv);
    v.setRandom();
    Eigen::VectorXd a(model.nv);
    a.setRandom();

    Eigen::VectorXd u = pinocchio::rnea(model, data, q, v, a);

    // Compute through function
    casadi::DM qd, vd, ad;
    damotion::utils::casadi::toCasadi(q, qd);
    damotion::utils::casadi::toCasadi(v, vd);
    damotion::utils::casadi::toCasadi(a, ad);

    casadi::DM ud = rnea(casadi::DMVector({qd, vd, ad}))[0];

    Eigen::VectorXd uc;
    damotion::utils::casadi::toEigen(ud, uc);

    EXPECT_TRUE(u.isApprox(uc));
}

TEST(PinocchioModelWrapper, EndEffector) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, false);
    pinocchio::Data data(model);

    damotion::utils::casadi::PinocchioModelWrapper wrapper(model);

    wrapper.addEndEffector("tool0");

    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model.nv);
    Eigen::VectorXd a = Eigen::VectorXd::Zero(model.nv);

    // Create function wrapper for end-effector function
    damotion::utils::casadi::FunctionWrapper ee(wrapper.end_effector(0).x);
    ee.setInput(0, q);
    ee.setInput(1, v);
    ee.setInput(2, a);

    ee.call();

    std::cout << ee.getOutput(0) << std::endl;
    std::cout << ee.getOutput(1) << std::endl;
    std::cout << ee.getOutput(2) << std::endl;

    EXPECT_TRUE(true);
}

TEST(PinocchioModelWrapper, RNEAWithEndEffector) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, false);
    pinocchio::Data data(model);

    damotion::utils::casadi::PinocchioModelWrapper wrapper(model);

    wrapper.addEndEffector("tool0");
    Eigen::Matrix<double, 6, 3> S;
    S.setZero();
    S.topRows(3).setIdentity();

    std::cout << S << std::endl;

    casadi::Function rnea = wrapper.rnea();

    std::cout << rnea << std::endl;

    EXPECT_TRUE(true);
}

TEST(PinocchioModelWrapper, PoseError) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, false);
    pinocchio::Data data(model);

    damotion::utils::casadi::PinocchioModelWrapper wrapper(model);

    wrapper.addEndEffector("tool0");

    damotion::utils::casadi::FunctionWrapper fee(
        damotion::utils::casadi::codegen(wrapper.end_effector(0).x, "./tmp")),
        f_err(
            damotion::utils::casadi::codegen(wrapper.end_effector(0).pose_error, "./tmp"));

    // Get two random configurations and compute transforms
    Eigen::VectorXd q0 = pinocchio::randomConfiguration(model),
                    q1 = pinocchio::randomConfiguration(model);

    pinocchio::forwardKinematics(model, data, q0);
    pinocchio::framesForwardKinematics(model, data, q0);
    pinocchio::SE3 se3_0 = data.oMf[model.getFrameId("tool0")];

    pinocchio::forwardKinematics(model, data, q1);
    pinocchio::framesForwardKinematics(model, data, q1);
    pinocchio::SE3 se3_1 = data.oMf[model.getFrameId("tool0")];

    // Compute log of the difference
    Eigen::Vector<double, 6> e_true =
        pinocchio::log6(se3_0.actInv(se3_1)).toVector();

    // Determine code-generated error
    Eigen::Quaterniond q(se3_1.rotation());
    Eigen::Vector3d x(se3_1.translation());

    f_err.setInput({0, 1, 2}, {q0, x, q.coeffs()});
    f_err.call();

    Eigen::Vector<double, 6> e = f_err.getOutput(0);

    EXPECT_TRUE(e.isApprox(e_true));
}