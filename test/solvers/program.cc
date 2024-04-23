#define DAMOTION_USE_PROFILING
#include "optimisation/program.h"

#include <gtest/gtest.h>

#include "solvers/qpoases.h"
#include "solvers/sparse.h"

namespace sym = damotion::symbolic;
namespace opt = damotion::optimisation;

TEST(Program, AddVariables) {
    // Create codegen function
    sym::VariableVector x = sym::CreateVariableVector("x", 2);
    sym::VariableVector y = sym::CreateVariableVector("y", 3);

    opt::Program program;

    program.AddDecisionVariables(x);
    program.AddDecisionVariables(y);

    program.ListDecisionVariables();

    EXPECT_TRUE(true);
}

TEST(Program, CreateOptimisationVector) {
    // Create codegen function
    sym::VariableVector x = sym::CreateVariableVector("x", 2);
    sym::VariableVector y = sym::CreateVariableVector("y", 2);
    sym::VariableVector z = sym::CreateVariableVector("z", 2);

    opt::Program program;

    program.AddDecisionVariables(x);
    program.AddDecisionVariables(y);
    program.AddDecisionVariables(z);

    sym::VariableVector opt(program.NumberOfDecisionVariables());
    opt << x, x, z;
    EXPECT_FALSE(program.SetDecisionVariableVector(opt));

    opt << y, x, z;
    EXPECT_TRUE(program.SetDecisionVariableVector(opt));

    // Check number of variables and parameters
    EXPECT_EQ(program.NumberOfParameters(), 0);
    EXPECT_EQ(program.NumberOfConstraints(), 0);
    EXPECT_EQ(program.NumberOfDecisionVariables(), 6);

    // Check indexing
    EXPECT_EQ(program.GetDecisionVariableIndex(y[0]), 0);
    EXPECT_EQ(program.GetDecisionVariableIndex(y[1]), 1);
    EXPECT_EQ(program.GetDecisionVariableIndex(x[0]), 2);
    EXPECT_EQ(program.GetDecisionVariableIndex(x[1]), 3);
}

TEST(Program, CreateParameters) {
    // Create codegen function
    opt::Program program;
    sym::Parameter a("a", 2, 3);
    program.AddParameter(a);

    EXPECT_EQ(a.rows(), 2);
    EXPECT_EQ(a.cols(), 3);

    Eigen::MatrixXd b(2, 3);
    b.setRandom();

    program.SetParameterValues(a, b);

    EXPECT_TRUE(program.GetParameterValues(a).isApprox(b));

    Eigen::MatrixXd c = program.GetParameterValues(a);
    EXPECT_TRUE(c.isApprox(b));
}

TEST(Program, AddLinearConstraint) {
    // Create codegen function
    sym::VariableVector x = sym::CreateVariableVector("x", 2);
    sym::VariableVector y = sym::CreateVariableVector("y", 2);

    opt::Program program;

    // Create constraint x0 + 2 y1 + 3 = 0
    Eigen::Matrix<double, 1, 2> A;
    A << 1.0, 2.0;
    Eigen::Vector<double, 1> b(3.0);

    std::shared_ptr<opt::LinearConstraint<Eigen::MatrixXd>> con =
        std::make_shared<opt::LinearConstraint<Eigen::MatrixXd>>(
            "", A, b, opt::BoundsType::kEquality);

    program.AddDecisionVariables(x);
    program.AddDecisionVariables(y);

    sym::VariableVector xy(2);
    xy << x[0], y[1];

    program.AddLinearConstraint(con, {xy}, {});
    LOG(INFO) << "Added Linear Constraint";

    sym::Parameter a("a", 2);
    program.AddParameter(a);

    LOG(INFO) << "Added Parameter";

    // Create random cost
    casadi::SX xx = casadi::SX::sym("x", 4);
    sym::Expression J = dot(xx, xx) + xx(0) + xx(0) * xx(2);
    J.SetInputs({xx}, {});
    std::shared_ptr<opt::QuadraticCost<Eigen::MatrixXd>> cost =
        std::make_shared<opt::QuadraticCost<Eigen::MatrixXd>>("sum_squares", J);

    sym::VariableVector xxyy(4);
    xxyy << x, y;
    program.AddQuadraticCost(cost, {xxyy}, {});

    LOG(INFO) << "Added Quadratic Cost";

    // Create optimisation vector
    program.SetDecisionVariableVector();

    program.AddBoundingBoxConstraint(-1.0, 1.0, x);
    program.AddBoundingBoxConstraint(-2.0, 2.0, y);

    LOG(INFO) << "Added Bounding Box Constraints";

    program.ListDecisionVariables();
    program.ListParameters();
    program.ListCosts();
    program.ListConstraints();

    // Create QPOASES solver and test if constraint jacobian gets created
    opt::solvers::QPOASESSolverInstance solver(program);

    LOG(INFO) << "Created solver";

    solver.Solve();
    solver.Solve();

    damotion::common::Profiler summary;
}

TEST(Program, SparseProgram) {
    // Create codegen function
    sym::VariableVector x = sym::CreateVariableVector("x", 10);

    opt::SparseProgram program;

    Eigen::Matrix<double, 1, 10> A1, A2;
    A1.setZero();
    A2.setZero();
    
    A1[2] = 1.0;
    A1[9] = -1.0;
    A2[5] = 1.0;
    A2[9] = -1.0;
    Eigen::Vector<double, 1> b1(1.0), b2(-2.0);

    std::shared_ptr<opt::LinearConstraint<Eigen::SparseMatrix<double>>>
        con1 = std::make_shared<
            opt::LinearConstraint<Eigen::SparseMatrix<double>>>(
            "", A1, b1, opt::BoundsType::kEquality),
        con2 = std::make_shared<
            opt::LinearConstraint<Eigen::SparseMatrix<double>>>(
            "", A2, b2, opt::BoundsType::kEquality);

    program.AddDecisionVariables(x);

    program.AddLinearConstraint(con1, {x}, {});
    program.AddLinearConstraint(con2, {x}, {});

    casadi::SX xx = casadi::SX::sym("x", 10);
    sym::Expression J = dot(xx, xx) + xx(0) + xx(0) * xx(2);
    J.SetInputs({xx}, {});
    std::shared_ptr<opt::QuadraticCost<Eigen::SparseMatrix<double>>> cost =
        std::make_shared<opt::QuadraticCost<Eigen::SparseMatrix<double>>>(
            "sum_squares", J);

    std::cout << cost->A() << std::endl;
    std::cout << cost->A().nonZeros() << std::endl;
    std::cout << cost->b() << std::endl;
    std::cout << cost->c() << std::endl;

    program.AddQuadraticCost(cost, {x}, {});

    // Create optimisation vector
    program.SetDecisionVariableVector();

    program.AddBoundingBoxConstraint(-10.0, 10.0, x);

    program.ListDecisionVariables();
    program.ListParameters();
    program.ListCosts();
    program.ListConstraints();

    opt::solvers::SparseSolver solver(program);

    damotion::common::Profiler summary;
}