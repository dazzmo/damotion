#define DAMOTION_USE_PROFILING
#include "solvers/program.h"

#include <gtest/gtest.h>

#include "solvers/solve_qpoases.h"

namespace sym = damotion::symbolic;
namespace opt = damotion::optimisation;

TEST(Program, AddVariables) {
    // Create codegen function
    sym::VariableVector x = sym::CreateVariableVector("x", 2);
    sym::VariableVector y = sym::CreateVariableVector("y", 3);

    opt::Program program;

    program.AddDecisionVariables(x);
    program.AddDecisionVariables(y);

    program.ListVariables();

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

    auto a = program.AddParameters("a", 2, 3);

    EXPECT_EQ(a.rows(), 2);
    EXPECT_EQ(a.cols(), 3);

    Eigen::MatrixXd b(2, 3);
    b.setRandom();

    program.SetParameters("a", b);

    EXPECT_TRUE(a.isApprox(b));

    Eigen::MatrixXd c = program.GetParameters("a");
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

    std::shared_ptr<opt::LinearConstraint> con =
        std::make_shared<opt::LinearConstraint>(
            A, b, opt::BoundsType::kEquality, "lin");

    program.AddDecisionVariables(x);
    program.AddDecisionVariables(y);

    sym::VariableVector xy(2);
    xy << x[0], y[1];

    program.AddLinearConstraint(con, {xy}, {});

    program.AddParameters("a", 2);

    // Create random cost
    casadi::SX xx = casadi::SX::sym("x", 4);
    sym::Expression J = dot(xx, xx) + xx(0) + xx(1);
    J.SetInputs({xx}, {});
    std::shared_ptr<opt::QuadraticCost> cost =
        std::make_shared<opt::QuadraticCost>(J, "sum_squares");

    sym::VariableVector xxyy(4);
    xxyy << x, y;
    program.AddQuadraticCost(cost, {xxyy}, {});

    // Create optimisation vector
    program.SetDecisionVariableVector();

    program.AddBoundingBoxConstraint(-1.0, 1.0, x);
    program.AddBoundingBoxConstraint(-2.0, 2.0, y);

    program.UpdateBindings();

    program.ListVariables();  // ! Name this decision variables
    program.ListParameters();
    program.ListCosts();
    program.ListConstraints();

    // opt::solvers::SolverBase solver(program);

    // Create QPOASES solver and test if constraint jacobian gets created
    opt::solvers::QPOASESSolverInstance solver(program);

    solver.Solve();
    solver.Solve();

    damotion::common::Profiler summary;
}