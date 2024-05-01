#ifndef SYMBOLIC_EXPRESSION_H
#define SYMBOLIC_EXPRESSION_H

#include "damotion/utils/codegen.h"
#include "damotion/utils/eigen_wrapper.h"

#include <casadi/casadi.hpp>
#include <cassert>


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
     * @brief Generates a function that evaluates the provided expression using
     * the provided inputs
     *
     * @param name
     * @param codegen
     * @param codegen_dir
     */
    void GenerateFunction(const std::string &name, bool codegen = false,
                          const std::string &codegen_dir = "./") {
        assert((x_.size() > 0 || p_.size() > 0) &&
               "No variables or parameters provided to expression!");

        if (codegen) {
            ::casadi::Function fcg =
                utils::casadi::codegen(toFunction(name), codegen_dir);
            f_ = utils::casadi::FunctionWrapper(fcg);
        } else {
            f_ = utils::casadi::FunctionWrapper(toFunction(name));
        }
    }

    utils::casadi::FunctionWrapper &Function() { return f_; }

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

   private:
    ::casadi::SXVector x_;
    ::casadi::SXVector p_;

    // Function to compute the expression
    utils::casadi::FunctionWrapper f_;

    // Create function from expression
    ::casadi::Function toFunction(const std::string &name) {
        // Create input
        ::casadi::SXVector in = this->Variables();
        for (const ::casadi::SX &pi : this->Parameters()) {
            in.push_back(pi);
        }
        return ::casadi::Function(name, in, {*this});
    }
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
     * @brief Generates a function that evaluates the provided expression
     * using the provided inputs
     *
     * @param name
     * @param codegen
     * @param codegen_dir
     */
    void GenerateFunction(const std::string &name, bool codegen = false,
                          const std::string &codegen_dir = "./") {
        assert((x_.size() > 0 || p_.size() > 0) &&
               "No variables or parameters provided to expression!");

        if (codegen) {
            ::casadi::Function fcg =
                utils::casadi::codegen(toFunction(name), codegen_dir);
            f_ = utils::casadi::FunctionWrapper(fcg);
        } else {
            f_ = utils::casadi::FunctionWrapper(toFunction(name));
        }
    }

    utils::casadi::FunctionWrapper &Function() { return f_; }

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

   private:
    ::casadi::SXVector x_;
    ::casadi::SXVector p_;

    // Function to compute the expression
    utils::casadi::FunctionWrapper f_;

    // Create function from expression
    ::casadi::Function toFunction(const std::string &name) {
        // Create input
        ::casadi::SXVector in = this->Variables();
        for (const ::casadi::SX &pi : this->Parameters()) {
            in.push_back(pi);
        }
        return ::casadi::Function(name, in, *this);
    }
};

}  // namespace symbolic
}  // namespace damotion

#endif /* SYMBOLIC_EXPRESSION_H */
