#ifndef SYMBOLIC_EXPRESSION_H
#define SYMBOLIC_EXPRESSION_H

#include <casadi/casadi.hpp>
#include <cassert>

#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"

namespace damotion {
namespace symbolic {

/**
 * @brief A symbolic expression (effectively a wrapper around a casadi::SX
 * object), that also includes the ability to specify the independent variables
 * and parameters within the expression.
 *
 */
class Expression : public ::casadi::SX {
   public:
    Expression() = default;
    ~Expression() = default;

    /**
     * @brief Copy constructor from an ::casadi::SX object
     *
     * @param other
     */
    Expression(const ::casadi::SX &other) : ::casadi::SX(other) {}

    void SetInputs(const ::casadi::SXVector &variables,
                   const ::casadi::SXVector &parameters) {
        x_ = variables;
        p_ = parameters;
    }

    /**
     * @brief Variables used within the expression
     *
     * @return ::casadi::SXVector&
     */
    const ::casadi::SXVector &Variables() const { return x_; }

    /**
     * @brief Parameters used within the expression
     *
     * @return ::casadi::SXVector&
     */
    const ::casadi::SXVector &Parameters() const { return p_; }

    /**
     * @brief Create a casadi::Function from the expression
     *
     * @param name
     * @return
     */
    ::casadi::Function toFunction(const std::string &name) {
        // Create input
        ::casadi::SXVector in = this->Variables();
        for (const ::casadi::SX &pi : this->Parameters()) {
            in.push_back(pi);
        }
        return ::casadi::Function(name, in, {*this});
    }

   private:
    ::casadi::SXVector x_;
    ::casadi::SXVector p_;
};

/**
 * @brief A vector of symbolic expressions (effectively a wrapper around a
 * casadi::SXVector object), that also includes the ability to specify the
 * independent variables and parameters within the expression.
 *
 */
class ExpressionVector : public ::casadi::SXVector {
   public:
    ExpressionVector() = default;
    ~ExpressionVector() = default;

    /**
     * @brief Copy constructor from an ::casadi::SXVector object
     *
     * @param other
     */
    ExpressionVector(const ::casadi::SXVector &other)
        : ::casadi::SXVector(other) {}

    void SetInputs(const ::casadi::SXVector &variables,
                   const ::casadi::SXVector &parameters) {
        x_ = variables;
        p_ = parameters;
    }

    /**
     * @brief Variables used within the expression
     *
     * @return ::casadi::SXVector&
     */
    ::casadi::SXVector &Variables() { return x_; }
    const ::casadi::SXVector &Variables() const { return x_; }

    /**
     * @brief Parameters used within the expression
     *
     * @return ::casadi::SXVector&
     */
    ::casadi::SXVector &Parameters() { return p_; }
    const ::casadi::SXVector &Parameters() const { return p_; }

    /**
     * @brief Create a casadi::Function from the expression
     *
     * @param name
     * @return
     */
    ::casadi::Function toFunction(const std::string &name) {
        // Create input
        ::casadi::SXVector in = this->Variables();
        for (const ::casadi::SX &pi : this->Parameters()) {
            in.push_back(pi);
        }
        return ::casadi::Function(name, in, *this);
    }

   private:
    ::casadi::SXVector x_;
    ::casadi::SXVector p_;
};

}  // namespace symbolic
}  // namespace damotion

#endif /* SYMBOLIC_EXPRESSION_H */
