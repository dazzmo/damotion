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
  Function() { Resize(0, 0, 0); }

  /**
   * @brief Construct a new Function object with number of variables nx and
   * number of outputs ny.
   *
   * @param nx Number of variables
   * @param ny Number of outputs
   * @param np Number of paramters
   */
  Function(const int nx, const int ny, const int& np = 0) {}

  ~Function() = default;

  /**
   * @brief Number of inputs for the function
   *
   * @return int
   */
  int nx() const { return nx_; }

  /**
   * @brief Number of parameters for the function
   *
   * @return int
   */
  int np() const { return np_; }

  /**
   * @brief Number of outputs for the function
   *
   * @return int
   */
  int ny() const { return ny_; }

  /**
   * @brief Resize the function to the appropriate outputs
   *
   * @param nx
   * @param ny
   * @param np
   */
  void Resize(const int& nx, const int& ny, const int& np = 0) {
    assert(nx > 0 && "A positive integer amount of inputs are required");
    assert(ny > 0 && "A positive integer amount of outputs are required");
    assert(np >= 0 &&
           "A non-negative integer amount of parameters are required");
    nx_ = nx;
    ny_ = ny;
    np_ = np;
    // Set number of inputs to the function (variables, parameters and
    // multipliers for hessian evaluation)
    in_.assign(nx_ + np_ + ny_, nullptr);
  }

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
   * @brief Set the i-th parameter for the function using the vector parameter.
   * Optionally can perform a check on the parameter to assess if it is valid.
   *
   * @param i
   * @param parameter
   * @param check
   */
  void SetParameter(const int& i,
                    const Eigen::Ref<const Eigen::VectorXd>& parameter,
                    bool check = false);

  /**
   * @brief Sets a collection of paramters to the function using the vector of
   * vector references parameter. Optionally can perform a check on the
   * parameter to assess if it is valid.
   *
   * @param indices
   * @param parameter
   * @param check
   */
  void SetParameter(
      const std::vector<int>& indices,
      const std::vector<Eigen::Ref<const Eigen::VectorXd>>& parameter,
      bool check = false);

  /**
   * @brief Set the i-th multipler for the function-multiplier product \f$
   * \lambda^T f \f$ for evaluation of the system hessian.
   *
   * @param i
   * @param parameter
   * @param check
   */
  void SetMultiplier(const int& i,
                     const Eigen::Ref<const Eigen::VectorXd>& multiplier,
                     bool check = false);

  /**
   * @brief Set the i-th multipler for the function-multiplier product \f$
   * \lambda^T f \f$ for evaluation of the system hessian. Optionally can
   * perform a check on the multiplier to assess if it is valid.
   *
   * @param indices
   * @param multiplier
   * @param check
   */
  void SetMultiplier(
      const std::vector<int>& indices,
      const std::vector<Eigen::Ref<const Eigen::VectorXd>>& multiplier,
      bool check = false);

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
  virtual const GenericEigenMatrix& GetDerivative(const int& i) const {
    throw std::runtime_error("Function does not have derivative information");
  }

  /**
   * @brief The hessian of the function with respect the inputs of the system
   * (concatenated as a single vector in the order of their inputs)
   *
   * @return const GenericEigenMatrix&
   */
  virtual const GenericEigenMatrix& GetHessian(const int& i) const {
    throw std::runtime_error("Function does not have hessian information");
  }

 protected:
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

  // Input data pointers that are stored for evaluation (x, p and optionally l)
  std::vector<const double*> in_;

  bool has_derivative_ = false;
  bool has_hessian_ = false;

 private:
  // Number of inputs
  int nx_;
  // Number of parameters
  int np_;
  // Number of outputs
  int ny_;
};

}  // namespace common
}  // namespace damotion

#endif /* COMMON_FUNCTION_H */
