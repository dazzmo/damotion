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
    EXPECT_EQ(program.GetDecisionVariableStartIndex(y), 0);
    EXPECT_EQ(program.GetDecisionVariableStartIndex(x), 2);
    EXPECT_EQ(program.GetDecisionVariableStartIndex(z), 4);
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

    Eigen::Matrix2d A;
    Eigen::Vector2d b;
    A.setRandom();
    b.setRandom();

    std::cout << A << std::endl;
    std::cout << b << std::endl;

    std::shared_ptr<opt::LinearConstraint> con =
        std::make_shared<opt::LinearConstraint>(
            A, b, opt::BoundsType::kEquality, "lin");

    program.AddDecisionVariables(x);
    program.AddDecisionVariables(y);

    program.AddLinearConstraint(con, {x}, {});
    program.AddLinearConstraint(con, {y}, {});

    program.AddParameters("a", 2);

    // Create optimisation vector
    program.SetDecisionVariableVector();

    program.AddBoundingBoxConstraint(-1.0, 1.0, x);

    program.UpdateBindings();

    program.ListVariables();  // ! Name this decision variables
    program.ListParameters();
    program.ListCosts();
    program.ListConstraints();

    // Create QPOASES solver and test if constraint jacobian gets created
    opt::solvers::QPOASESSolverInstance solver(program);

    solver.Solve();
}