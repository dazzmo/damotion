#ifndef CORE_FUNCTION_HPP
#define CORE_FUNCTION_HPP

#include <Eigen/Core>

#include "damotion/core/fwd.hpp"
#include "damotion/core/optional_matrix.hpp"

namespace damotion {

#define OptionalNone static_cast<OptionalMatrix>(nullptr)

// Helper trait to calculate the number of lower triangular block matrices
// needed
template <std::size_t N>
struct LowerTriangularBlockMatricesCount {
  static constexpr std::size_t value = N * (N + 1) / 2;
};

template <std::size_t N>
struct JacobianBlockMatricesCount {
  static constexpr std::size_t value = N;
};

template <std::size_t, typename T>
using alwaysT = T;

template <typename ReturnValue, typename InputSequence,
          typename HessianSequence>
class FunctionBaseImpl;

template <typename ReturnValue, std::size_t... InputSequence,
          std::size_t... HessianSequence>
class FunctionBaseImpl<ReturnValue, std::index_sequence<InputSequence...>,
                       std::index_sequence<HessianSequence...>> {
 public:
  using Index = std::size_t;

  using ReturnType = ReturnValue;
  using LagrangeMultiplierType = ReturnValue;

  using VectorType = Eigen::VectorXd;

  using JacobianType =
      std::conditional_t<std::is_same<ReturnValue, double>::value,
                         Eigen::RowVectorXd, Eigen::MatrixXd>;

  using HessianType = Eigen::MatrixXd;

  // Input types

  using InputVectorType = VectorType;

  // Optional input types

  using OptionalVectorType = OptionalVector;

  using OptionalJacobianType =
      std::conditional_t<std::is_same<ReturnValue, double>::value,
                         OptionalRowVector, OptionalMatrix>;

  using OptionalHessianType = OptionalMatrix;

  FunctionBaseImpl() = default;
  ~FunctionBaseImpl() = default;

  /**
   * @brief Evaluate the function and optionally, the Jacobians of the output
   * with respect to the input values
   *
   * @param val
   * @param jac
   */
  virtual ReturnType evaluate(
      const alwaysT<InputSequence, InputVectorType> &...in,
      alwaysT<InputSequence, OptionalJacobianType>... jac) const = 0;

  /**
   * @brief Function to compute
   *
   * @param val
   * @param lam
   * @param hes
   */
  virtual void hessian(
      const alwaysT<InputSequence, InputVectorType> &...in,
      const InputVectorType &lam,
      alwaysT<HessianSequence, OptionalHessianType>... hes) const {}

  const InputVectorType &get_parameters() const { return p_; }
  void set_parameters(const InputVectorType &p) { p_ = p; }

 protected:
 private:
  // Parameters for the function
  InputVectorType p_;
};

/**
 * @brief Class for any function which can be evaluated with N inputs, and
 * optionally compute their Jacobians and Hessians.
 *
 * @tparam N Number of inputs to the function
 * @tparam ReturnType Return type of the function
 * @tparam InputType Input type for the function
 */
template <std::size_t N, typename ReturnType = Eigen::VectorXd>
using FunctionBase = FunctionBaseImpl<
    ReturnType, std::make_index_sequence<N>,
    std::make_index_sequence<LowerTriangularBlockMatricesCount<N>::value>>;

// bool checkInput(const InputRef& v) {
//   if (v.hasNaN() || !v.allFinite()) {
//     std::ostringstream ss;
//     LOG(ERROR) << "Input has invalid values:\n" << v.transpose().format(3);
//     throw std::runtime_error(ss.str());
//   }
//   return true;
// }

}  // namespace damotion

#endif /* CORE_FUNCTION_HPP */
