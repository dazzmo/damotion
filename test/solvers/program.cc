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
    Eigen::Matrix<double, 1, 2> Ax, Ay;
    Ax << 1.0, 0.0;
    Ay << 0.0, 2.0;
    Eigen::Vector<double, 1> b(3.0);

    std::shared_ptr<opt::LinearConstraint> con =
        std::make_shared<opt::LinearConstraint>(
            std::vector<Eigen::MatrixXd>({Ax, Ay}), b,
            opt::BoundsType::kEquality, "lin");

    program.AddDecisionVariables(x);
    program.AddDecisionVariables(y);

    program.AddLinearConstraint(con, {x, y}, {});

    program.AddParameters("a", 2);

    // Create random cost
    casadi::SX xx = casadi::SX::sym("x", 2);
    casadi::SX yy = casadi::SX::sym("y", 2);

    sym::Expression J = xx(0) + yy(0);
    J.SetInputs({xx, yy}, {});

    std::shared_ptr<opt::Cost> c =
        std::make_shared<opt::Cost>(J, "sum_squares", true, true);

    program.AddCost(c, {x, y}, {});

    // Create optimisation vector
    program.SetDecisionVariableVector();

    program.AddBoundingBoxConstraint(-1.0, 1.0, x);

    program.UpdateBindings();

    program.ListVariables();  // ! Name this decision variables
    program.ListParameters();
    program.ListCosts();
    program.ListConstraints();

    // Create QPOASES solver and test if constraint jacobian gets created
    // opt::solvers::QPOASESSolverInstance solver(program);

    // solver.Solve();
}