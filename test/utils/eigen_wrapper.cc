#include "utils/eigen_wrapper.h"

#include <gtest/gtest.h>
#include "common/logging.h"
#include "utils/codegen.h"

TEST(EigenWrapper, EigenWrapperLoad) {
    // Create codegen function
    casadi::SX x = casadi::SX::sym("x"), y = casadi::SX::sym("y");
    casadi::Function f("test", {x, y}, {x + y}, {"x", "y"}, {"l"});

    damotion::utils::casadi::FunctionWrapper wrapper(f);

    // Evaluate function
    Eigen::VectorXd x_in(1), y_in(1), output(1);
    x_in.setRandom();
    y_in.setRandom();
    LOG(INFO) << "Calling";
    wrapper.call({x_in, y_in});
    LOG(INFO) << "Getting Output";
    output = wrapper.getOutput(0);

    EXPECT_DOUBLE_EQ(output[0], (x_in[0] + y_in[0]));
}

TEST(EigenWrapper, ToCasadiDM) {
    // Create codegen function
    int n = 10;
    Eigen::VectorXd x(n);
    x.setRandom();

    // Convert to casadi::DM
    casadi::DM xd;
    damotion::utils::casadi::toCasadi(x, xd);

    // Convert back to Eigen::VectorXd
    Eigen::VectorXd xt;
    damotion::utils::casadi::toEigen(xd, xt);

    EXPECT_TRUE(x.isApprox(xt));
}

TEST(EigenWrapper, EigenWrapperSparse) {
    // Create codegen function
    casadi::SX x = casadi::SX::sym("x");
    casadi::SX y(2, 2);
    y(0, 0) = 1.0;
    y(1, 1) = 1.0;

    casadi::Function f("sparse_test", {x}, {y}, {"x"}, {"y"});

    damotion::utils::casadi::FunctionWrapper wrapper(
        damotion::utils::casadi::codegen(f, "./tmp"));

    // Evaluate function
    Eigen::VectorXd x_in(1);
    x_in.setRandom();

    Eigen::Matrix2d I;
    I.setIdentity();

    wrapper.setSparseOutput(0);
    wrapper.call({x_in});
    Eigen::SparseMatrix<double> res = wrapper.getOutputSparse(0);

    EXPECT_TRUE(res.isApprox(I));
}

TEST(EigenWrapper, EigenWrapperCodegenLoad) {
    // Create codegen function
    casadi::SX x = casadi::SX::sym("x"), y = casadi::SX::sym("y");
    casadi::Function f("test", {x, y}, {x + y}, {"x", "y"}, {"l"});

    damotion::utils::casadi::FunctionWrapper wrapper(
        damotion::utils::casadi::codegen(f, "./tmp"));

    // Evaluate function
    Eigen::VectorXd x_in(1), y_in(1), output(1);
    x_in.setRandom();
    y_in.setRandom();

    wrapper.call({x_in, y_in});
    output = wrapper.getOutput(0);

    EXPECT_DOUBLE_EQ(output[0], (x_in[0] + y_in[0]));
}