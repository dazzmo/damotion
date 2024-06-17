#ifndef COMMON_FUNCTION_H
#define COMMON_FUNCTION_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cassert>
#include <functional>
#include <iostream>

#include "damotion/common/eigen.h"
#include "damotion/common/logging.h"
#include "damotion/common/sparsity.h"

namespace damotion {

typedef double Scalar;

typedef std::vector<const Scalar*> InputDataVector;
typedef std::vector<Scalar*> OutputDataVector;

typedef Eigen::Ref<Eigen::VectorX<Scalar>> VectorRef;
typedef Eigen::Ref<Eigen::MatrixX<Scalar>> MatrixRef;
typedef Eigen::Ref<Eigen::SparseMatrix<Scalar>> SparseMatrixRef;
typedef Eigen::Ref<const Eigen::VectorX<Scalar>> ConstVectorRef;
typedef Eigen::Ref<const Eigen::MatrixX<Scalar>> ConstMatrixRef;

namespace common {

/**
 * @brief Base class for a function \f$ y = f(x) \f$ that takes independent
 * variables x and returns the outputs y.
 *
 */
class Function {
 public:
  using UniquePtr = std::unique_ptr<Function>;
  using SharedPtr = std::shared_ptr<Function>;

  /**
   * @brief Empty constructor for the Function class
   *
   */
  Function() : n_in_(0), n_out_(0), size_in_({}) {}

  /**
   * @brief Construct a new Function object with number of variables n_in and
   * number of outputs n_out.
   *
   * @param n_in Number of inputs
   * @param n_out Number of outputs
   */
  Function(const size_t& n_in, const size_t& n_out)
      : n_in_(n_in), n_out_(n_out) {}

  ~Function() = default;

  /**
   * @brief Number of inputs for the function
   *
   * @return size_t
   */
  size_t n_in() const { return n_in_; }

  /**
   * @brief Number of outputs for the function
   *
   * @return size_t
   */
  size_t n_out() const { return n_out_; }

  /**
   * @brief Size of the i-th input
   *
   * @param i
   * @return size_t
   */
  size_t size_in(const size_t& i) const {
    assert(i < n_in() && "Index for input out of range");
    return size_in_[i];
  }

  /**
   * @brief Evaluates the function using the provided inputs
   *
   * @param in Vector of inputs to evaluate the function
   * @param out Vector of outputs to evaluate to. If the vector is shorter than
   * @ref n_out(), only the provided outputs will be evaluated.
   * @param check Whether to assess each input for inconsistencies (e.g.
   * infinite values, bad data)
   */
  void eval(const InputDataVector& in, OutputDataVector& out,
            bool check = false) {
    assert(in.size() == n_in() && "Incorrect number of inputs provided");
    assert(out.size() == n_out() && "Incorrect number of outputs provided");
    // Perform input check, if using
    if (check) {
      for (const auto& x : in) {
        checkInput(x);
      }
    }
    // Evaluate the function using the provided implementation
    evalImpl(in, out);
  }

  /**
   * @brief Retrive the sparsity information for the i-th output of the
   * function. If the expression is dense, the inner and outer pointer vectors
   * will return with a size of zero.
   *
   * @param i The index of the input to extract
   * @param rows The number of rows for the input
   * @param cols The number of columns for the input
   * @param nnz The number of non-zero elements in the input
   * @param i_row Row data in triplet form
   * @param j_col Column data in triplet form
   */
  void getOutputSparsityInfo(const size_t& i, size_t& rows, size_t& cols,
                             size_t& nnz, std::vector<int>& i_row,
                             std::vector<int>& j_col) {
    assert(i < n_out() && "Number of outputs exceeded");
    getOutputSparsityInfoImpl(i, rows, cols, nnz, i_row, j_col);
  }

  // TODO - Reduced number of arguments for getOutputSparsityInfo

 protected:
  /**
   * @brief Virtual method for derived class to override
   *
   * @param input
   */
  virtual void evalImpl(const InputDataVector& in, OutputDataVector& out) = 0;

  virtual void getOutputSparsityInfoImpl(const size_t& i, size_t& rows,
                                         size_t& cols, size_t& nnz,
                                         std::vector<int>& inner,
                                         std::vector<int>& outer) = 0;

  bool checkInput(const ConstVectorRef& v) {
    if (v.hasNaN() || !v.allFinite()) {
      std::ostringstream ss;
      ss << "Input has invalid values:\n" << v.transpose().format(3);
      throw std::runtime_error(ss.str());
    }
    return true;
  }

 private:
  // Number of inputs
  size_t n_in_;
  // Number of outputs
  size_t n_out_;

  // Size of each input argument
  std::vector<size_t> size_in_;
};

}  // namespace common
}  // namespace damotion

#endif /* COMMON_FUNCTION_H */
