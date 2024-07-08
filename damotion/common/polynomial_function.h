#ifndef COMMON_POLYNOMIAL_FUNCTION_H
#define COMMON_POLYNOMIAL_FUNCTION_H

#include "damotion/common/function.h"

namespace damotion {
namespace common {

class PolynomialFunction {
 public:
  PolynomialFunction() = default;
  ~PolynomialFunction() = default;

  PolynomialFunction(const size_t& degree) : degree_(degree), fcoef_(nullptr) {}

  /**
   * @brief Sets the function that evaluates the coefficients of the polynomial
   * function.
   *
   * @param fcoef
   */
  void setCoefficientsFunction(const Function::SharedPtr& fcoef) {
    fcoef_ = std::move(fcoef);
  }

  const Function::SharedPtr& getCoefficientsFunction() const { return fcoef_; }
  Function::SharedPtr& getCoefficientsFunction() { return fcoef_; }

  /**
   * @brief Whether a coefficients function exists to evaluate the coefficients
   * of the polynomial.
   *
   * @return true
   * @return false
   */
  bool hasCoefficientsFunction() const { return fcoef_ != nullptr; }

 private:
  size_t degree_;
  Function::SharedPtr fcoef_;
};

}  // namespace common
}  // namespace damotion

#endif /* COMMON_POLYNOMIAL_FUNCTION_H */
