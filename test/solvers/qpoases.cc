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
    std::size_t n = 10;
    // Symbolic cost creation
    sym s = sym::sym("s", n);

    dopt::Cost::SharedPtr obj =
        std::make_shared<dcas::QuadraticCost>("qc", sym::dot(s, s), s);

    // Add constraint
    dopt::LinearConstraint::SharedPtr con =
        std::make_shared<dcas::LinearConstraint>("lc", s(0) + 1, s);
    con->setBoundsFromType(dopt::BoundType::STRICTLY_NEGATIVE);

    dopt::LinearConstraint::SharedPtr con2 =
        std::make_shared<dcas::LinearConstraint>("lc", s(0) - s(1) - 2.0, s);
    con->setBoundsFromType(dopt::BoundType::STRICTLY_POSITIVE);

    // Create variables
    dsym::Vector x = dsym::createVector("x", n);

    program.x().add(x);
    // Create objective
    program.f().add(obj, x, {});
    program.g().add(con, x, {});
    program.g().add(con2, x, {});
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

  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;
  FLAGS_v = 1;

  int status = RUN_ALL_TESTS();
  
  damotion::Profiler summary;
  return status;
}