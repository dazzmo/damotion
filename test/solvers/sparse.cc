
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "damotion/casadi/function.hpp"
#include "damotion/core/logging.hpp"
#include "damotion/core/profiler.hpp"
#include "damotion/optimisation/program.h"
#include "damotion/solvers/qpoases.h"
#include "damotion/symbolic/variable.hpp"

using sym = ::casadi::SX;
using dm = ::casadi::DM;

namespace dcas = damotion::casadi;
namespace dsym = damotion::symbolic;
namespace dopt = damotion::optimisation;

TEST(qpoases, SparseObjective) {
  std::size_t n = 2000;
  // Symbolic cost creation
  sym s = sym::sym("s", n);

  dopt::Cost::SharedPtr obj =
      std::make_shared<dcas::Cost>("qc", s(0) * s(100) * s(20) + s(500), s);

  // Compute gradient of the cost
  Eigen::VectorXd x(n);
  Eigen::MatrixXd J(1, n);

  J.setZero();
  for (std::size_t i = 0; i < 5; ++i) {
    x.setRandom();
    Eigen::RowVectorXd grd(n);
    obj->evaluate(x, grd);
    J += grd;
  }

  // Get sparse approximation
  Eigen::SparseMatrix<double> J_sparse = J.sparseView(0.0, 1e-3);
  J_sparse.makeCompressed();
  std::cout << J_sparse << '\n';

  // Fill components of a sparse matrix

  // todo - have utility that iterates over the sparse matrix in an iterator
  // todo - fashion

  // provide matrix and only update if it's in the block?

  for (int i = 0; i < 1000; i++) {
    damotion::Profiler("iterator");
    for (int k = 0; k < J_sparse.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(J_sparse, k); it;
           ++it) {
        it.valueRef() = J(it.row(), it.col());
      }
    }
  }

  Eigen::MatrixXd block(1, 4);
  block.setRandom();
  std::vector<std::size_t> rows = {0};
  std::vector<std::size_t> cols = {0, 100, 20, 500};

  for (int i = 0; i < 1000; i++) {
    damotion::Profiler("proposed");
    updateBlock(J_sparse, block, rows, cols);
  }
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