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
 * @brief Base class for a function \f$ y = f(x, p) \f$ that takes independent
 * variables x and parameters p and returns the outputs y.
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
  Function() {
    SetNumberOfInputs(0);
    SetNumberOfOutputs(0);
  }

  /**
   * @brief Construct a new Function object with number of inputs n_x and
   * number of outputs n_y.
   *
   * @param n_x Number of inputs
   * @param n_y Number of outputs
   */
  Function(const int n_x, const int n_y, const int& n_p = 0) {
    SetNumberOfInputs(n_x);
    SetNumberOfInputs(n_p);
    SetNumberOfOutputs(n_y);
  }

  ~Function() = default;

  /**
   * @brief Number of inputs for the function
   *
   * @return const int&
   */
  const int& n_x() const { return n_x_; }

  /**
   * @brief Number of parameters for the function
   *
   * @return const int&
   */
  const int& n_p() const { return n_p_; }

  /**
   * @brief Number of outputs for the function
   *
   * @return const int&
   */
  const int& n_y() const { return n_y_; }

  /**
   * @brief Calls the function, with options to compute the derivative and
   * hessian of the function if implemented.
   *
   * @param derivative
   * @param hessian
   * @return * void
   */
  void call(const bool derivative = false, const bool hessian = false) {
    EvalImpl();
    if (derivative) DerivativeImpl();
    if (hessian) HessianImpl();
  }

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
  void SetInput(const std::vector<int>& indices,
                const std::vector<Eigen::Ref<const Eigen::VectorXd>>& input,
                bool check = false);

  /**
   * @brief Set the i-th input for the function using the data array input.
   * Optionally can perform a check on the input to assess if it is valid.
   *
   * @param i
   * @param input
   * @param check
   */
  void SetInput(const int& i, const double* input, bool check = false);

  /**
   * @brief Sets a collection of inputs to the function using the vector of data
   * pointers input. Optionally can perform a check on the input to assess if it
   * is valid.
   *
   * @param indices
   * @param input
   * @param check
   */
  void SetInput(const std::vector<int>& indices,
                const std::vector<const double*> input, bool check = false);

  /**
   * @brief Set the i-th paramter for the function using the vector paramter.
   * Optionally can perform a check on the paramter to assess if it is valid.
   *
   * @param i
   * @param paramter
   * @param check
   */
  void SetParameter(const int& i,
                    const Eigen::Ref<const Eigen::VectorXd>& paramter,
                    bool check = false);

  /**
   * @brief Sets a collection of paramters to the function using the vector of
   * vector references paramter. Optionally can perform a check on the paramter
   * to assess if it is valid.
   *
   * @param indices
   * @param paramter
   * @param check
   */
  void SetParameter(
      const std::vector<int>& indices,
      const std::vector<Eigen::Ref<const Eigen::VectorXd>>& paramter,
      bool check = false);

  /**
   * @brief Set the i-th paramter for the function using the data array
   * paramter. Optionally can perform a check on the paramter to assess if it is
   * valid.
   *
   * @param i
   * @param paramter
   * @param check
   */
  void SetParameter(const int& i, const double* paramter, bool check = false);

  /**
   * @brief Sets a collection of inputs to the function using the vector of data
   * pointers input. Optionally can perform a check on the input to assess if it
   * is valid.
   *
   * @param indices
   * @param input
   * @param check
   */
  void SetParameter(const std::vector<int>& indices,
                    const std::vector<const double*> input, bool check = false);

  /**
   * @brief Returns the i-th output as a GenericEigenMatrix object.
   *
   * @param i
   * @return const GenericEigenMatrix&
   */
  virtual const GenericEigenMatrix& GetOutput(const int& i) const = 0;

  /**
   * @brief Whether the function has the ability to provide its first
   * derivative, accessed by GetDerivative()
   *
   * @return true
   * @return false
   */
  bool HasDerivative() const { return has_derivative_; }

  /**
   * @brief Whether the function has the ability to provide its second
   * derivative (hessian), accessed by GetDerivative()
   *
   * @return true
   * @return false
   */
  bool HasHessian() const { return has_hessian_; }

  /**
   * @brief The derivative of the function with respect the inputs of the system
   * (concatenated as a single vector in the order of their inputs)
   *
   * @param i
   * @return const GenericEigenMatrix&
   */
  virtual const GenericEigenMatrix& GetDerivative() const {
    throw std::runtime_error("Function " + this->name() +
                             " does not have derivative information");
  }

  /**
   * @brief The hessian of the function with respect the inputs of the system
   * (concatenated as a single vector in the order of their inputs)
   *
   * @return const GenericEigenMatrix&
   */
  virtual const GenericEigenMatrix& GetHessian() const {
    throw std::runtime_error("Function " + this->name() +
                             " does not have hessian information");
  }

 protected:
  /**
   * @brief Set the number of inputs the function has.
   *
   * @param n
   */
  void SetNumberOfInputs(const int& n) {
    assert(n > 0 && "A positive integer amount of inputs are required");
    n_x_ = n;
    in_.assign(n_x_, nullptr);
  }

  /**
   * @brief Set the number of parameters the function has.
   *
   * @param n
   */
  void SetNumberOfParameters(const int& n) {
    assert(n > 0 && "A non-negative integer amount of parameters are required");
    n_p_ = n;
    in_.assign(n_p_, nullptr);
  }

  /**
   * @brief Set the number of outputs the function has
   *
   * @param n
   */
  void SetNumberOfOutputs(const int& n) {
    assert(n > 0 && "A positive integer amount of outputs are required");
    n_y_ = n;
  }

  /**
   * @brief Virtual method for derived class to override
   *
   * @param input
   */
  virtual void EvalImpl() = 0;

  /**
   * @brief Virtual method for derivative information to be taken from for the
   * function.
   *
   */
  virtual void DerivativeImpl() = 0;

  /**
   * @brief Virtual method for hessian information to be taken from for the
   * function.
   *
   */
  virtual void HessianImpl() = 0;

  // Input data pointers that are stored for evaluation
  std::vector<const double*> in_;

 private:
  // Number of inputs
  int n_x_;
  // Number of parameters
  int n_p_;
  // Number of outputs
  int n_y_;

  bool has_derivative_;
  bool has_hessian_;
};

}  // namespace common
}  // namespace damotion

#endif /* COMMON_FUNCTION_H */
