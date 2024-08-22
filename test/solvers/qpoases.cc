#define DAMOTION_USE_PROFILING

#include "damotion/solvers/qpoases.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/casadi/function.hpp"
#include "damotion/core/logging.hpp"
#include "damotion/core/profiler.hpp"
#include "damotion/optimisation/program.h"
#include "damotion/symbolic/variable.hpp"

using sym = ::casadi::SX;
using dm = ::casadi::DM;

namespace dcas = damotion::casadi;
namespace dsym = damotion::symbolic;
namespace dopt = damotion::optimisation;

class BasicObjective {
 public:
  BasicObjective() : program() {
    // Symbolic cost creation
    sym s = sym::sym("s", 2);
    // Binomial objective
    sym f = pow(s(0) + s(1), 2);

    // Create variables
    dsym::Variable x("x"), y("y");

    dopt::Cost::SharedPtr obj =
        std::make_shared<dcas::QuadraticCost>("qc", f, s);

    // Add constraint
    f = s(0) + s(1) - 1;
    dopt::LinearConstraint::SharedPtr con =
        std::make_shared<dcas::LinearConstraint>("lc", f, s);
    con->setBoundsFromType(dopt::BoundType::STRICTLY_POSITIVE);

    dsym::Vector vec(2);
    vec << x, y;

    program.x().add(x);
    program.x().add(y);
    // Create objective
    program.f().add(obj, vec, {});
    program.g().add(con, vec, {});
  }

  dopt::MathematicalProgram program;

 private:
};

TEST(qpoases, BasicObjective) {
  BasicObjective problem;
  dopt::solvers::QPOASESSolverInstance qp(problem.program);

  // Attempt to solve the program
  qp.solve();

  std::cout << qp.getPrimalSolution() << '\n';
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  int status = RUN_ALL_TESTS();
  damotion::Profiler summary;
  return status;
}