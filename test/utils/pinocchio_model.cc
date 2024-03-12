#include "utils/pinocchio_model.h"

#include <gtest/gtest.h>

#include "pinocchio/parsers/urdf.hpp"
#include "utils/codegen.h"

using namespace casadi_utils::eigen;

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

    casadi_utils::PinocchioModelWrapper wrapper(model);

    casadi::Function aba = wrapper.aba();

    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v(model.nv);
    v.setRandom();
    Eigen::VectorXd tau(model.nv);
    tau.setRandom();

    Eigen::VectorXd a = pinocchio::aba(model, data, q, v, tau);

    // Compute through function
    casadi::DM qd, vd, ud;
    toCasadi(q, qd);
    toCasadi(v, vd);
    toCasadi(tau, ud);

    casadi::DM ad = aba(casadi::DMVector({qd, vd, ud}))[0];

    Eigen::VectorXd ac;
    toEigen(ad, ac);

    EXPECT_TRUE(a.isApprox(ac));
}

TEST(PinocchioModelWrapper, ABATestCodegen) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, false);
    pinocchio::Data data(model);

    casadi_utils::PinocchioModelWrapper wrapper(model);

    casadi::Function aba = wrapper.aba();
    aba = casadi_utils::codegen(aba, "./tmp/");

    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v(model.nv);
    v.setRandom();
    Eigen::VectorXd tau(model.nv);
    tau.setRandom();

    Eigen::VectorXd a = pinocchio::aba(model, data, q, v, tau);

    // Compute through function
    casadi::DM qd, vd, ud;
    toCasadi(q, qd);
    toCasadi(v, vd);
    toCasadi(tau, ud);

    casadi::DM ad = aba(casadi::DMVector({qd, vd, ud}))[0];

    Eigen::VectorXd ac;
    toEigen(ad, ac);

    EXPECT_TRUE(a.isApprox(ac));
}

TEST(PinocchioModelWrapper, RNEATest) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, false);
    pinocchio::Data data(model);

    casadi_utils::PinocchioModelWrapper wrapper(model);

    casadi::Function rnea = wrapper.rnea();

    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v(model.nv);
    v.setRandom();
    Eigen::VectorXd a(model.nv);
    a.setRandom();

    Eigen::VectorXd u = pinocchio::rnea(model, data, q, v, a);

    // Compute through function
    casadi::DM qd, vd, ad;
    toCasadi(q, qd);
    toCasadi(v, vd);
    toCasadi(a, ad);

    casadi::DM ud = rnea(casadi::DMVector({qd, vd, ad}))[0];

    Eigen::VectorXd uc;
    toEigen(ud, uc);

    EXPECT_TRUE(u.isApprox(uc));
}

TEST(PinocchioModelWrapper, RNEATestCodegen) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, false);
    pinocchio::Data data(model);

    casadi_utils::PinocchioModelWrapper wrapper(model);

    casadi::Function rnea = wrapper.rnea();
    rnea = casadi_utils::codegen(rnea, "./tmp/");

    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v(model.nv);
    v.setRandom();
    Eigen::VectorXd a(model.nv);
    a.setRandom();

    Eigen::VectorXd u = pinocchio::rnea(model, data, q, v, a);

    // Compute through function
    casadi::DM qd, vd, ad;
    toCasadi(q, qd);
    toCasadi(v, vd);
    toCasadi(a, ad);

    casadi::DM ud = rnea(casadi::DMVector({qd, vd, ad}))[0];

    Eigen::VectorXd uc;
    toEigen(ud, uc);

    EXPECT_TRUE(u.isApprox(uc));
}

TEST(PinocchioModelWrapper, EndEffector) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, false);
    pinocchio::Data data(model);

    casadi_utils::PinocchioModelWrapper wrapper(model);

    wrapper.addEndEffector("tool0");

    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model.nv);
    Eigen::VectorXd a = Eigen::VectorXd::Zero(model.nv);
    
    // Create function wrapper for end-effector function
    casadi_utils::eigen::FunctionWrapper ee(wrapper.end_effector(0).x);
    ee.setInput(0, q);
    ee.setInput(1, v);
    ee.setInput(2, a);
    
    ee.call();
    
    std::cout << ee.getOutput(0) << std::endl;
    std::cout << ee.getOutput(1) << std::endl;
    std::cout << ee.getOutput(2) << std::endl;
    
    // Evaluate Jacobian at nominal configuration
    ee_jac.setSparseOutput(0);
    ee_jac.setInput(0, q);

    ee_jac.call();
    std::cout << ee_jac.getOutputSparse(0) << std::endl;

    EXPECT_TRUE(true);
}

TEST(PinocchioModelWrapper, RNEAWithEndEffector) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, false);
    pinocchio::Data data(model);

    casadi_utils::PinocchioModelWrapper wrapper(model);

    wrapper.addEndEffector("tool0");
    Eigen::Matrix<double, 6, 3> S;
    S.setZero();
    S.topRows(3).setIdentity();

    std::cout << S << std::endl;

    casadi::Function rnea = wrapper.rnea();

    std::cout << rnea << std::endl;

    EXPECT_TRUE(true);
}