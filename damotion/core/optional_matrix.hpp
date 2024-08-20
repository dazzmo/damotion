#ifndef OPTIONAL_MATRIX_HPP
#define OPTIONAL_MATRIX_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <optional>

namespace damotion {

/**
 * @brief A class representing an optional matrix argument for a function,
 which
 * can be included optionally or ignored by using setting the default value
 to
 * its null constructor.
 *
 * @tparam Rows
 * @tparam Cols
 *
 */
template <int Rows, int Cols>
class OptionalMatrixData {
 public:
  typedef Eigen::Matrix<double, Rows, Cols> MatrixType;

  OptionalMatrixData() : map_(nullptr, Rows, Cols) {}

  OptionalMatrixData(std::nullptr_t) : map_(nullptr, Rows, Cols) {}
  OptionalMatrixData(std::nullopt_t) : map_(nullptr, Rows, Cols) {}

  // Create from structured matrix
  template <int U, int V>
  OptionalMatrixData(Eigen::Matrix<double, U, V>& M)
      : map_(nullptr, Rows, Cols) {
    new (&map_) Eigen::Map<MatrixType>(M.data(), M.rows(), M.cols());
  }

  // Create from structured matrix
  template <int U, int V>
  OptionalMatrixData(OptionalMatrixData<U, V>& M) : map_(nullptr, Rows, Cols) {
    new (&map_) Eigen::Map<MatrixType>(M->data(), M->rows(), M->cols());
  }

  OptionalMatrixData(double& val) : map_(nullptr, Rows, Cols) {
    new (&map_) Eigen::Map<MatrixType>(&val, 1, 1);
  }

  operator bool() const { return map_.data() != nullptr; }

  Eigen::Map<MatrixType>* operator->() { return &map_; }
  Eigen::Map<MatrixType>& operator*() { return map_; }

  const Eigen::Map<MatrixType>* operator->() const { return &map_; }
  const Eigen::Map<MatrixType>& operator*() const { return map_; }

 private:
  Eigen::Map<MatrixType> map_;
};

using OptionalVector = OptionalMatrixData<-1, 1>;
using OptionalRowVector = OptionalMatrixData<1, -1>;

// template <>
// class OptionalMatrixData<-1, -1> {
//  public:
//   typedef Eigen::MatrixXd MatrixType;

//   OptionalMatrixData() : map_(nullptr, 1, 1) {}

//   OptionalMatrixData(std::nullptr_t) : map_(nullptr, 1, 1) {}
//   OptionalMatrixData(std::nullopt_t) : map_(nullptr, 1, 1) {}

//   // Create from structured matrix
//   template <int Rows, int Cols>
//   OptionalMatrixData(Eigen::Matrix<double, Rows, Cols>& M)
//       : map_(nullptr, 1, 1) {
//     new (&map_) Eigen::Map<MatrixType>(M.data(), M.rows(), M.cols());
//   }

//   // Create from structured matrix
//   template <int Rows, int Cols>
//   OptionalMatrixData(OptionalMatrixData<Rows, Cols>& M) : map_(nullptr, 1, 1)
//   {
//     new (&map_) Eigen::Map<MatrixType>(M->data(), M->rows(), M->cols());
//   }

//   operator bool() const { return map_.data() != nullptr; }

//   Eigen::Map<MatrixType>* operator->() { return &map_; }
//   Eigen::Map<MatrixType>& operator*() { return map_; }

//   const Eigen::Map<MatrixType>* operator->() const { return &map_; }
//   const Eigen::Map<MatrixType>& operator*() const { return map_; }

//  private:
//   Eigen::Map<MatrixType> map_;
// };

using OptionalMatrix = OptionalMatrixData<-1, -1>;

// using OptionalVector = OptionalDynamicMatrix<-1, 1>;
// using OptionalRowVector = OptionalDynamicMatrix<1, -1>;
// using OptionalMatrix = OptionalDynamicMatrix<-1, -1>;

}  // namespace damotion
#endif /* OPTIONAL_MATRIX_HPP */
