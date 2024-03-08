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