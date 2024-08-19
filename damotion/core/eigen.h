#ifndef CORE_EIGEN_H
#define CORE_EIGEN_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>

#include "damotion/core/fwd.hpp"

namespace damotion {

/**
 * @brief Create a Eigen::SparseMatrix of size rows x cols using the triplet
 * data and optional value initialisation. Returns a matrix in compressed form.
 *
 * @param rows
 * @param cols
 * @param i_row
 * @param j_col
 * @param val
 * @return Eigen::SparseMatrix<double>
 */
Eigen::SparseMatrix<double> getSparseMatrixFromTripletData(
    const std::size_t& rows, const std::size_t& cols, const std::size_t& nnz,
    const std::vector<int>& i_row, const std::vector<int>& j_col,
    const std::vector<double>& val = {}) {
  // Compute sparse matrix through eigen
  std::vector<Eigen::Triplet<int>> triplets;
  double v = 0.0;
  for (std::size_t i = 0; i < nnz; ++i) {
    if (val.size()) v = val[i];
    triplets.push_back(Eigen::Triplet<int>(i_row[i], j_col[i], v));
  }
  Eigen::SparseMatrix<double> mat(rows, cols);
  mat.setFromTriplets(triplets.begin(), triplets.end());
  mat.makeCompressed();
  return mat;
}

/**
 * @brief Creates an Eigen::SparseMatrix from a dense matrix M
 *
 * @param M
 * @return Eigen::SparseMatrix<double>
 */
Eigen::SparseMatrix<double> toSparse(
    const Eigen::Ref<const Eigen::MatrixXd>& M) {
  return M.sparseView();
}

/**
 * @brief Sets the values of an Eigen::SparseMatrix from its dense equivalent,
 * by iterating through the non-zero elements of the sparse matrix.
 *
 * @param res
 * @param val
 */
void setSparse(Eigen::SparseMatrix<double>& res,
               const Eigen::Ref<const Eigen::MatrixXd>& val) {
  // Iterate over non-zero elements
  for (std::size_t k = 0; k < res.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(res, k); it; ++it) {
      it.valueRef() = val(it.row(), it.col());
    }
  }
}

}  // namespace damotion

#endif/* CORE_EIGEN_H */
