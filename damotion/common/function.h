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
  Function() { Resize(0, 0, 0); }

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
   * @return int
   */
  int n_in() const { return n_in_; }

  /**
   * @brief Number of outputs for the function
   *
   * @return int
   */
  int n_out() const { return n_out_; }

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
    // Set number of inputs to the function (variables, parameters and
    // multipliers for hessian evaluation)
    in_.assign(n_in_, nullptr);
  }

  /**
   * @brief Calls the function
   *
   */
  void call() { EvalImpl(); }

  /**
   * @brief Set the i-th input for the function using the vector input.
   * Optionally can perform a check on the input to assess if it is valid.
   *
   * @param i
   * @param input
   * @param check
   */
  void SetInput(const int& i, const Eigen::Ref<const Eigen::VectorXd>& input,
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
  void SetInputs(const std::vector<int>& indices,
                 const std::vector<Eigen::Ref<const Eigen::VectorXd>>& input,
                 bool check = false);

  /**
   * @brief Returns the i-th output as a GenericEigenMatrix object.
   *
   * @param i
   * @return const GenericEigenMatrix&
   */
  virtual const GenericEigenMatrix& GetOutput(const int& i) const = 0;

 protected:
  /**
   * @brief Virtual method for derived class to override
   *
   * @param input
   */
  virtual void EvalImpl() = 0;

  // Input data pointers that are stored for evaluation (x, p and optionally l)
  std::vector<const double*> in_;

 private:
  // Number of inputs
  int n_in_;
  // Number of outputs
  int n_out_;
};

}  // namespace common
}  // namespace damotion

#endif /* COMMON_FUNCTION_H */
