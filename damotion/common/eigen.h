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
Eigen::SparseMatrix<double> SparseMatrixFromTripletData(
    const size_t& rows, const size_t& cols, const std::vector<int>& i_row,
    const std::vector<int>& j_col, const std::vector<double>& val = {}) {
  // Compute sparse matrix through eigen
  std::vector<Eigen::Triplet<int>> triplets;
  size_t nnz = i_row.size();
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

/**
 * @brief This class represents any matrix-type data of Eigen by representing
 it
 * as an array of continuous data. At its core, it is a sparse matrix object
 * handled by Eigen's SparseMatrix class. The data vector can then be viewed
 as
 * any of Eigen's MatrixBase objects, such dense vectors and matrices as well
 as
 * of course sparse matrices. The data array is initialised on construction
 of
 * the class, making the array sufficiently large to represent its intended
 * matrix type. Any view can be chosen by this class provided the data array
 is
 * large enough to represent the desired view.
 *
 */
class GenericEigenMatrix {
 public:
  using SharedPtr = std::shared_ptr<GenericEigenMatrix>;
  using UniquePtr = std::unique_ptr<GenericEigenMatrix>;

  GenericEigenMatrix() = default;
  ~GenericEigenMatrix() = default;

  /**
   * @brief Construct a new GenericEigenMatrix object for the 1x1 matrix
   class.
   *
   */
  GenericEigenMatrix(const double& val) {
    data_.reserve(1);
    data_.coeffRef(0, 0) = val;
    // Convert matrix to compressed form
    data_.makeCompressed();
  }

  /**
   * @brief Construct a new GenericEigenMatrix object for a dense matrix with
   * structure given by mat. Copies the data of mat to the GenericEigenMatrix
   * object.
   *
   * @param mat
   */
  GenericEigenMatrix(const int& rows, const int& cols) {
    data_.reserve(rows * cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        data_.coeffRef(i, j) = 0.0;
      }
    }
    // Convert matrix to compressed form
    data_.makeCompressed();
  }

  /**
   * @brief Construct a new GenericEigenMatrix object for a dense matrix with
   * structure given by mat. Copies the data of mat to the GenericEigenMatrix
   * object.
   *
   * @param mat
   */
  GenericEigenMatrix(const Eigen::Ref<const Eigen::MatrixXd>& mat) {
    data_.reserve(mat.rows() * mat.cols());
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < mat.cols(); ++j) {
        data_.coeffRef(i, j) = mat(i, j);
      }
    }
    // Convert matrix to compressed form
    data_.makeCompressed();
  }

  /**
   * @brief Construct a new Generic Matrix Data object by copying a Eigen
   * SparseMatrix object with its data and sparsity pattern.
   *
   * @param mat
   */
  GenericEigenMatrix(const Eigen::Ref<const Eigen::SparseMatrix<double>>& mat) {
    data_ = mat;
  }

  /**
   * @brief Express the GenericEigenMatrix as a scalar value.
   *
   * @return double
   */
  double toDouble() {
    assert(data_.nonZeros() == 1 && "Data array has more than one value");
    return data_.valuePtr()[0];
  }

  const double& toConstDoubleRef() const {
    assert(data_.nonZeros() == 1 && "Data array has more than one value");
    return data_.valuePtr()[0];
  }

  Eigen::Map<const Eigen::VectorXd> toConstVectorXdRef() const {
    return Eigen::Map<const Eigen::VectorXd>(data_.valuePtr(),
                                             data_.nonZeros());
  }

  Eigen::Map<Eigen::VectorXd> toVectorXdRef() {
    return Eigen::Map<Eigen::VectorXd>(data_.valuePtr(), data_.nonZeros());
  }

  Eigen::VectorXd toVectorXd() {
    return Eigen::Map<Eigen::VectorXd>(data_.valuePtr(), data_.nonZeros());
  }

  /**
   * @brief Returns a constant reference to a mapped Eigen::MatrixXd object
   * based on the data of the GenericMatrixData
   *
   * @return Eigen::Map<const Eigen::MatrixXd>
   */
  Eigen::Map<const Eigen::MatrixXd> toConstMatrixXdRef() const {
    assert(data_.nonZeros() == data_.rows() * data_.cols() &&
           "Data array is not large to provide a matrix map");
    return Eigen::Map<const Eigen::MatrixXd>(data_.valuePtr(), data_.rows(),
                                             data_.cols());
  }

  /**
   * @brief Create an Eigen::MatrixXd object from the GenericMatrixData
   object.
   *
   * @return Eigen::MatrixXd
   */
  Eigen::MatrixXd toMatrixXd() {
    // TODO - Convert to dense matrix instead
    assert(data_.nonZeros() == data_.rows() * data_.cols() &&
           "Data array is not large to provide a matrix map");
    return Eigen::Map<const Eigen::MatrixXd>(data_.valuePtr(), data_.rows(),
                                             data_.cols());
  }

  /**
   * @brief Returns a copy of the SparseMatrix class that manages the data
   array
   * for the GenericEigenMatrix class.
   *
   * @return Eigen::SparseMatrix<double>
   */
  Eigen::SparseMatrix<double> toSparseMatrix() const { return data_; }

  /**
   * @brief Returns a reference to the SparseMatrix class that manages the
   data
   * array for the GenericEigenMatrix class.
   *
   * @return Eigen::SparseMatrix<double>&
   */
  Eigen::Ref<Eigen::SparseMatrix<double>> toSparseMatrixRef() { return data_; }

  /**
   * @brief Returns a reference to the SparseMatrix class that manages the
   data
   * array for the GenericEigenMatrix class.
   *
   * @return Eigen::SparseMatrix<double>&
   */
  Eigen::Ref<const Eigen::SparseMatrix<double>> toConstSparseMatrixRef() const {
    return data_;
  }

  /**
   * @brief Returns the pointer to the data array for the GenericEigenMatrix
   * object. This can be used to modify and insert values directly into the
   data
   * array.
   *
   * @return double*
   */
  double* data() { return data_.valuePtr(); }

  /**
   * @brief The number of non-zero elements in the data array for the
   * GenericEigenMatrix class.
   *
   * @return const Eigen::Index&
   */
  Eigen::Index nnz() { return data_.nonZeros(); }

 private:
  Eigen::SparseMatrix<double> data_;
};

std::ostream& operator<<(std::ostream& os, const GenericEigenMatrix& data) {
  std::ostringstream oss;
  oss << data.toConstSparseMatrixRef() << '\n';
  return os << oss.str();
}

class MatrixData {
 public:
  Eigen::Ref<const Eigen::MatrixXd> asDense() {
    return Eigen::Map<Eigen::MatrixXd>(data.data(), rows, cols);
  }

  Eigen::Ref<const Eigen::SparseMatrix<double>> asSparse() {
    return Eigen::Map<Eigen::SparseMatrix<double>>(
        nnz, rows, cols, inner.data(), outer.data(), data.data());
  }

 private:
  int rows;
  int cols;
  int nnz;
  std::vector<int> inner;
  std::vector<int> outer;
  std::vector<double> data;
};

}  // namespace damotion

#endif /* COMMON_EIGEN_DATA_H */
