#include "damotion/solvers/base.h"

namespace damotion {
namespace optimisation {
namespace solvers {

void updateSparseMatrix(Eigen::SparseMatrix<double> &M,
                        const Eigen::MatrixXd &block,
                        const std::vector<std::size_t> &row_indices,
                        const std::vector<std::size_t> &col_indices,
                        const Operation &operation) {
  for (int k = 0; k < M.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
      auto p_row = std::find(row_indices.begin(), row_indices.end(), it.row());
      if (p_row != row_indices.end()) {
        auto p_col =
            std::find(col_indices.begin(), col_indices.end(), it.col());
        if (p_col != col_indices.end()) {
          const double &val =
              block(p_row - row_indices.begin(), p_col - col_indices.begin());
          if (operation == Operation::SET) {
            it.valueRef() = val;
          } else if (operation == Operation::ADD) {
            it.valueRef() += val;
          }
        }
      }
    }
  }
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion