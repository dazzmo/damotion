#ifndef COMMON_SPARSITY_H
#define COMMON_SPARSITY_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <casadi/casadi.hpp>
#include <vector>

namespace damotion {
namespace common {

class Sparsity {
 public:
  Sparsity() : rows_(1), cols_(1), nnz_(1) {}
  ~Sparsity() = default;

  Sparsity(const long long nrows, const long long ncols,
           const std::vector<long long> &vrows = {},
           const std::vector<long long> &vcols = {})
      : rows_(nrows), cols_(ncols), vrows_(vrows), vcols_(vcols) {}

  Sparsity(const Eigen::SparseMatrix<double> &M) {
    // Copy sparsity pattern
    rows_ = M.rows();
    cols_ = M.cols();
    nnz_ = M.nonZeros();
    // Create triplet data
    vrows_ = {};
    vcols_ = {};
    std::vector<Eigen::Triplet<double>> v;
    for (int i = 0; i < M.outerSize(); i++) {
      for (typename Eigen::SparseMatrix<double>::InnerIterator it(M, i); it;
           ++it) {
        vrows_.emplace_back(it.row());
        vcols_.emplace_back(it.col());
      }
    }
  }

  /**
   * @brief Construct a new Sparsity object based on a casadi::Sparsity object
   *
   * @param sparsity
   */
  Sparsity(const casadi::Sparsity &sparsity) {
    // Copy sparsity pattern
    rows_ = sparsity.rows();
    cols_ = sparsity.columns();
    nnz_ = sparsity.nnz();
    // Create triplet data
    vrows_ = {};
    vcols_ = {};
    sparsity.get_triplet(vrows_, vcols_);
  }

  const long long &rows() const { return rows_; }
  const long long &cols() const { return cols_; }
  const long long &nnz() const { return nnz_; }

  void GetTriplets(std::vector<long long> &rows, std::vector<long long> &cols) {
    rows = vrows_;
    cols = vcols_;
  }

 private:
  long long rows_ = 0;
  long long cols_ = 0;
  long long nnz_ = 0;

  std::vector<long long> vrows_;
  std::vector<long long> vcols_;
};

/**
 * @brief Create a Sparse Eigen Matrix object given a sparsity pattern
 *
 * @param sparsity
 * @return Eigen::SparseMatrix<double>
 */
Eigen::SparseMatrix<double> CreateSparseEigenMatrix(Sparsity &sparsity);

}  // namespace common
}  // namespace damotion

#endif /* COMMON_SPARSITY_H */
