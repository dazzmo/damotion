#ifndef UTILS_CASADI_H
#define UTILS_CASADI_H

#include <casadi/casadi.hpp>
#include <cassert>

namespace damotion {
namespace utils {
namespace casadi {
/**
 * @brief Creates the symbolic inputs of a function given the function that
 * possesses named inputs
 *
 * @param f
 * @return ::casadi::SXVector
 */
::casadi::StringVector CreateInputNames(::casadi::Function &f);

/**
 * @brief Creates a gradient function for expression f with outputs being
 * gradients with respect to the variables given by x.
 *
 * @param name Name of the expression
 * @param f Expression to generate the gradients for
 * @param in Symbolic inputs required to generate the expression
 * @param inames Names of the symbolic inputs in in
 * @param x Symbolic variables to compute the gradients with respect to
 * @param xnames Names of the symbolic variables in x
 * @return casadi::Function
 */
::casadi::Function CreateGradientFunction(const std::string &name,
                                          const ::casadi::SX &f,
                                          const ::casadi::SXVector &in,
                                          const ::casadi::StringVector &inames,
                                          const ::casadi::SXVector &x,
                                          const ::casadi::StringVector &xnames);

/**
 * @brief Creates a jacobian function for expression f with outputs being
 * jacobians with respect to the variables given by x.
 *
 * @param name Name of the expression
 * @param f Expression to generate the gradients for
 * @param in Symbolic inputs required to generate the expression
 * @param inames Names of the symbolic inputs in in
 * @param x Symbolic variables to compute the gradients with respect to
 * @param xnames Names of the symbolic variables in x
 * @return casadi::Function
 */
::casadi::Function CreateJacobianFunction(const std::string &name,
                                          const ::casadi::SX &f,
                                          const ::casadi::SXVector &in,
                                          const ::casadi::StringVector &inames,
                                          const ::casadi::SXVector &x,
                                          const ::casadi::StringVector &xnames);

/**
 * @brief Creates a hessian function for expression f with outputs being
 * hessians with respect to the variables given by x.
 *
 * @param name Name of the expression
 * @param f Expression to generate the gradients for
 * @param in Symbolic inputs required to generate the expression
 * @param inames Names of the symbolic inputs in in
 * @param xy Pairs of symbolic variables to compute the hessian of f with
 * respect to
 * @param xynames Pairs of names for the symbolic variables in xy
 * @return casadi::Function
 */
::casadi::Function CreateHessianFunction(
    const std::string &name, const ::casadi::SX &f,
    const ::casadi::SXVector &in, const ::casadi::StringVector &inames,
    const std::vector<std::pair<::casadi::SX, ::casadi::SX>> &xy,
    const std::vector<std::pair<std::string, std::string>> &xynames);

}  // namespace casadi
}  // namespace utils
}  // namespace damotion

#endif /* UTILS_CASADI_H */
