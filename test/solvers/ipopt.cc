
#include "damotion/solvers/ipopt.h"

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
    std::size_t n = 2;
    // Symbolic cost creation
    sym s = sym::sym("s", n);

    dopt::QuadraticCost::SharedPtr obj =
        std::make_shared<dcas::QuadraticCost>("qc", sym::dot(s, s), s);

    dopt::LinearConstraint::SharedPtr con =
        std::make_shared<dcas::LinearConstraint>("lc", s(0) - s(1), s);
    con->setBoundsFromType(dopt::BoundType::POSITIVE);

    // Create variables
    dsym::Vector x = dsym::createVector("x", n);

    program.x().add(x);
    program.x().initialise(x[0], 0.5);
    program.x().initialise(x[1], -0.5);
    // Create objective
    program.f().add(obj, x, {});
    // Add constraints
    program.g().add(con, x, {});
    program.g().addBoundingBoxConstraint("bb", x, -1.0, 1.0);

    LOG(INFO) << program.g();
  }

  dopt::MathematicalProgram program;

 private:
};

TEST(qpoases, BasicObjective) {
  BasicObjective problem;
  dopt::solvers::IpoptSolver nlp(problem.program);

  // Attempt to solve the program
  nlp.solve();
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);

  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;
  FLAGS_v = 0;

  int status = RUN_ALL_TESTS();

  damotion::Profiler summary;
  return status;
}