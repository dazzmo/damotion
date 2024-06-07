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
 * @brief Vector of input vector references to the function
 *
 */
typedef std::vector<Eigen::Ref<const Eigen::VectorXd>> InputRefVector;

class Function {
 public:
  using UniquePtr = std::unique_ptr<Function>;
  using SharedPtr = std::shared_ptr<Function>;

  /**
   * @brief Empty constructor for the Function class
   *
   */
  Function() {
    SetNumberOfInputs(0);
    SetNumberOfOutputs(0);
  }

  /**
   * @brief Construct a new Function object with number of inputs n_in and
   * number of outputs n_out.
   *
   * @param n_in Number of inputs
   * @param n_out Number of outputs
   */
  Function(const int n_in, const int n_out) {
    SetNumberOfInputs(n_in);
    SetNumberOfOutputs(n_out);
  }

  ~Function() = default;

  /**
   * @brief Number of inputs for the function
   *
   * @return const int&
   */
  const int &n_in() const { return n_in_; }

  /**
   * @brief Number of outputs for the function
   *
   * @return const int&
   */
  const int &n_out() const { return n_out_; }

  /**
   * @brief Call the function based on the current inputs set by SetInput()
   *
   */
  void call() { callImpl(); }

  /**
   * @brief Set the i-th input for the function using the vector input.
   * Optionally can perform a check on the input to assess if it is valid.
   *
   * @param i
   * @param input
   * @param check
   */
  void SetInput(const int &i, const Eigen::Ref<const Eigen::VectorXd> &input,
                bool check = false);

  /**
   * @brief Sets a collection of inputs to the function using the vector of
   * vector references input. Optionally can perform a check on the input to
   * assess if it is valid.
   *
   * @param indices
   * @param input
   * @param check
   */
  void SetInput(const std::vector<int> &indices,
                const std::vector<Eigen::Ref<const Eigen::VectorXd>> &input,
                bool check = false);

  /**
   * @brief Set the i-th input for the function using the data array input.
   * Optionally can perform a check on the input to assess if it is valid.
   *
   * @param i
   * @param input
   * @param check
   */
  void SetInput(const int &i, const double *input, bool check = false);

  /**
   * @brief Sets a collection of inputs to the function using the vector of data
   * pointers input. Optionally can perform a check on the input to assess if it
   * is valid.
   *
   * @param indices
   * @param input
   * @param check
   */
  void SetInput(const std::vector<int> &indices,
                const std::vector<const double *> input, bool check = false);

  /**
   * @brief Returns the current value of output i
   *
   * @param i
   * @return const GenericEigenMatrix&
   */
  const GenericEigenMatrix &getOutput(const int &i) const { return out_[i]; }

 protected:
  /**
   * @brief Set the number of inputs the function has.
   *
   * @param n
   */
  void SetNumberOfInputs(const int &n) {
    assert(n > 0 && "A positive integer amount of inputs are required");
    n_in_ = n;
    in_.assign(n_in_, nullptr);
  }

  /**
   * @brief Set the number of outputs the function has
   *
   * @param n
   */
  void SetNumberOfOutputs(const int &n) {
    assert(n > 0 && "A positive integer amount of outputs are required");
    n_out_ = n;
    out_.resize(n_out_);
  }

  /**
   * @brief Virtual method for derived class to override
   *
   * @param input
   */
  virtual void callImpl() = 0;

  // Input data pointers that are stored for evaluation
  std::vector<const double *> in_;

  /**
   * @brief Vector of output matrices
   *
   * @return std::vector<Eigen::MatrixXd>&
   */
  std::vector<GenericEigenMatrix> &OutputVector() { return out_; }

 private:
  int n_in_;
  int n_out_;

  // Output data vector of GenericEigenMatrix objects
  mutable std::vector<GenericEigenMatrix> out_;
};

/**
 * @brief Function that operates by callback
 *
 */
class CallbackFunction : public Function {
 public:
  typedef std::function<void(const std::vector<const double *> &,
                             std::vector<GenericEigenMatrix> &)>
      f_callback_;

  CallbackFunction() = default;
  ~CallbackFunction() = default;

  CallbackFunction(const int n_in, const int n_out, const f_callback_ &callback)
      : Function(n_in, n_out) {
    SetCallback(callback);
  }

  /**
   * @brief Initialise the output i provided its sparsity pattern
   *
   * @param i The index of the output to initialise
   * @param sparsity Sparsity object detailing the structure of the matrix
   */
  void InitialiseOutput(const int i, const Sparsity &sparsity) {
    // TODO - Fix this up
    // this->OutputVector()[i] =
    //     GenericEigenMatrix::Zero(sparsity.rows(), sparsity.cols());
  }

  /**
   * @brief Set the callback to be used when call() is used for the function.
   *
   * @param callback
   */
  void SetCallback(const f_callback_ &callback) { f_ = callback; }

  /**
   * @brief Calls the callback function provided to it
   *
   * @param input
   */
  void callImpl() override {
    if (f_ == nullptr) {
      throw std::runtime_error(
          "Function callback not provided to CallbackFunction!");
    }
    f_(in_, this->OutputVector());
  }

 private:
  f_callback_ f_ = nullptr;
};

}  // namespace common
}  // namespace damotion

#endif /* COMMON_FUNCTION_H */
