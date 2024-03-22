#ifndef SOLVERS_COST_H
#define SOLVERS_COST_H

#include <casadi/casadi.hpp>
#include "utils/eigen_wrapper.h"

namespace damotion {
namespace optimisation {

class Cost {
       public:
        Cost() = default;
        ~Cost() = default;

        Cost(const std::string &name) : name_(name) {}

        void SetSymbolicObjective(const casadi::SX &c) { c_ = c; }
        const casadi::SX &SymbolicObjective() const { return c_; }

        void SetSymbolicInputs(const casadi::SXVector &inputs) {
            inputs_ = inputs;
        }
        const casadi::SXVector &SymbolicInputs() const { return inputs_; }

        void SetObjectiveFunction(casadi::Function &f) { obj_ = f; }
        void SetGradientFunction(casadi::Function &f) { grad_ = f; }
        void SetHessianFunction(casadi::Function &f) { hes_ = f; }

        casadi_utils::eigen::FunctionWrapper &ObjectiveFunction() { return obj_; }
        casadi_utils::eigen::FunctionWrapper &GradientFunction() { return grad_; }
        casadi_utils::eigen::FunctionWrapper &HessianFunction() { return hes_; }

        /**
         * @brief Name of the cost
         *
         * @return const std::string&
         */
        const std::string &name() const { return name_; }

        /**
         * @brief Cost weighting
         *
         * @return const double
         */
        const double weighting() const { return w_; }

        /**
         * @brief Cost weighting
         *
         * @return double&
         */
        double &weighting() { return w_; }

       private:
        // Cost name
        std::string name_;
        // Cost weighting
        double w_;

        // Underlying symbolic representation of objective
        casadi::SX c_;
        // Symbolic input vector
        casadi::SXVector inputs_;

        /**
         * @brief Objective function
         *
         */
        casadi_utils::eigen::FunctionWrapper obj_;

        /**
         * @brief Gradient function
         *
         */
        casadi_utils::eigen::FunctionWrapper grad_;

        /**
         * @brief Hessian function
         *
         */
        casadi_utils::eigen::FunctionWrapper hes_;
    };


}
}

#endif/* SOLVERS_COST_H */
