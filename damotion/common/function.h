#ifndef COMMON_FUNCTION_H
#define COMMON_FUNCTION_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cassert>
#include <functional>
#include <iostream>

#include "damotion/common/eigen.h"
#include "damotion/common/logging.h"

namespace damotion {

typedef double Scalar;

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

  using Input = Eigen::Ref<const Eigen::VectorX<Scalar>>;
  using InputVector = std::vector<Input>;

  class Output {
   public:
    using DenseType = Eigen::MatrixX<Scalar>;
    using SparseType = Eigen::SparseMatrix<Scalar>;

    Output() = default;

    /**
     * @brief Creates a dense function output of dimensions rows x cols
     *
     * @param rows
     * @param cols
     */
    Output(const size_t& rows, const size_t& cols) : out_s_(nullptr) {
      out_d_ = std::make_shared<Eigen::MatrixXd>(rows, cols);
    }

    /**
     * @brief Construct a sparse matrix output using the provided sparsity
     * pattern.
     *
     * @param rows
     * @param cols
     * @param nnz
     * @param i_row
     * @param j_col
     */
    Output(size_t& rows, size_t& cols, size_t& nnz, std::vector<int>& i_row,
           std::vector<int>& j_col)
        : out_d_(nullptr) {
      out_s_ = std::make_shared<SparseType>(
          getSparseMatrixFromTripletData(rows, cols, nnz, i_row, j_col));
    }

    bool isSparse() const { return out_s_ != nullptr; }

    bool isDense() const { return out_d_ != nullptr; }

    /**
     * @brief Returns the output as a dense matrix
     *
     * @return const Eigen::MatrixXd&
     */
    const DenseType& asDense() const {
      assert(out_d_ && "Output is constructed as sparse");
      // Provide compatibility if sparse matrix is provided
      return *out_d_;
    }

    DenseType& asDense() {
      assert(out_d_ && "Output is constructed as sparse");
      // Provide compatibility if sparse matrix is provided
      return *out_d_;
    }

    const SparseType& asSparse() const {
      assert(out_s_ && "Output is constructed as dense");
      return *out_s_;
    }

    SparseType& asSparse() {
      assert(out_s_ && "Output is constructed as dense");
      return *out_s_;
    }

    /**
     * @brief Provides a pointer to the data array that defines the output.
     *
     * @return Scalar*
     */
    Scalar* data() {
      if (isDense()) return out_d_->data();
      if (isSparse()) return out_s_->valuePtr();
      // Otherwise, throw an error
      throw std::runtime_error("Output is not set for dense or sparse output!");
    }

   private:
    // Shared pointer to the sparse representation of the output
    std::shared_ptr<SparseType> out_s_ = nullptr;
    // Shared pointer to the dense representation of the output
    std::shared_ptr<DenseType> out_d_ = nullptr;
  };

  using OutputVector = std::vector<Output>;

  /**
   * @brief Returns the i-th output of the function as a Function::Output
   * object.
   *
   * @param i
   * @return const Output&
   */
  const Output& getOutput(const size_t& i) {
    assert(i < n_out() && "Number of outputs exceeded");
    return out_[i];
  }

  /**
   * @brief Empty constructor for the Function class
   *
   */
  Function() : n_in_(0), n_out_(0), size_in_({}) {}
  ~Function() = default;

  /**
   * @brief Construct a new Function object with number of variables n_in and
   * number of outputs n_out.
   *
   * @param n_in Number of inputs
   * @param n_out Number of outputs
   */
  Function(const size_t& n_in, const size_t& n_out)
      : n_in_(n_in), n_out_(n_out) {}

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
   * @brief Evaluates all outputs of the function using the inputs given be in
   *
   * @param in
   */
  void eval(const InputVector& in, bool check = false) {
    assert(in.size() == n_in() && "Incorrect number of inputs provided");
    // Perform input check, if using
    if (check)
      for (const auto& x : in) checkInput(x);
    // Evaluate
    evalImpl(in);
  }

 protected:
  /**
   * @brief Virtual method for derived class to override
   *
   * @param input
   */
  virtual void evalImpl(const InputVector& in) = 0;

  /**
   * @brief Perform checks on the provided input
   *
   * @param v
   * @return true
   * @return false
   */
  bool checkInput(const Input& v) {
    if (v.hasNaN() || !v.allFinite()) {
      std::ostringstream ss;
      ss << "Input has invalid values:\n" << v.transpose().format(3);
      throw std::runtime_error(ss.str());
    }
    return true;
  }

  /**
   * @brief Vector of function outputs
   *
   */
  std::vector<Output> out_;

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
