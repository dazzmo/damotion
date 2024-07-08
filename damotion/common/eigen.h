#ifndef COMMON_EIGEN_DATA_H
#define COMMON_EIGEN_DATA_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>

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
    const size_t& rows, const size_t& cols, const size_t& nnz, const std::vector<int>& i_row,
    const std::vector<int>& j_col, const std::vector<double>& val = {}) {
  // Compute sparse matrix through eigen
  std::vector<Eigen::Triplet<int>> triplets;
  double v = 0.0;
  for (size_t i = 0; i < nnz; ++i) {
    if (val.size()) v = val[i];
    triplets.push_back(Eigen::Triplet<int>(i_row[i], j_col[i], v));
  }
  Eigen::SparseMatrix<double> mat(rows, cols);
  mat.setFromTriplets(triplets.begin(), triplets.end());
  mat.makeCompressed();
  return mat;
}

}  // namespace damotion

#endif /* COMMON_EIGEN_DATA_H */
