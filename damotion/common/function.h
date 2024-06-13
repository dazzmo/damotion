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
namespace common {

/**
 * @brief Vector of Eigen::Ref<const Eigen::VectorXd>
 *
 */
typedef VectorRef Eigen::Ref<Eigen::VectorXd>;
typedef ConstVectorRef Eigen::Ref<const Eigen::VectorXd>;

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
  Function() { Resize(0, 0); }

  /**
   * @brief Construct a new Function object with number of variables n_in and
   * number of outputs n_out.
   *
   * @param n_in Number of variables
   * @param n_out Number of outputs
   */
  Function(const int n_in, const int n_out) {}

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
   * @brief Resize the function to the appropriate outputs
   *
   * @param n_in
   * @param n_out
   */
  void Resize(const int& n_in, const int& n_out) {
    assert(n_in > 0 && "A positive integer amount of inputs are required");
    assert(n_out > 0 && "A positive integer amount of outputs are required");
    n_in_ = n_in;
    n_out_ = n_out;
    // Set number of inputs to the function
    in_.assign(n_in_, nullptr);
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
  void Eval(const std::vector<Eigen::Ref<const Eigen::VectorXd>>& in,
            std::vector<Eigen::Ref<Eigen::MatrixXd>>& out, bool check = false) {
    assert(in.size() == n_in() && "Incorrect number of inputs provided");
    assert(out.size() == n_out() && "Incorrect number of outputs provided");
    // Perform input check, if using
    if (check) {
      for (const auto& x : in) {
        CheckInput(x);
      }
    }
    // Evaluate the function using the provided implementation
    EvalImpl(in, out);
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
  void GetOutputSparsityInfo(const size_t& i, size_t& rows, size_t& cols,
                             size_t& nnz, std::vector<int>& i_row,
                             std::vector<int>& j_col) {
    assert(out.size() < n_out() && "Number of outputs exceeded");
    GetOutputSparsityInfoImpl(i, rows, cols, nnz, i_row, j_col);
  }

 protected:
  /**
   * @brief Virtual method for derived class to override
   *
   * @param input
   */
  virtual void EvalImpl(
      const std::vector<const Eigen::Ref<Eigen::VectorXd>>& in,
      std::vector<Eigen::Ref<Eigen::MatrixXd>>& out) = 0;

  virtual void GetOutputSparsityInfoImpl(const size_t& i, size_t& rows,
                                         size_t& cols, size_t& nnz,
                                         std::vector<int>& inner,
                                         std::vector<int>& outer) = 0;

  bool CheckInput(const Eigen::Ref<const Eigen::VectorXd>& v) {
    if (v.hasNaN() || !v.allFinite()) {
      std::ostringstream ss;
      ss << "Input has invalid values:\n" << v.transpose().format(3);
      throw std::runtime_error(ss.str());
    }
    return true;
  }

  void CheckInputRange(const size_t& i) {}

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
