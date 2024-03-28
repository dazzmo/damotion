#ifndef SOLVERS_COST_H
#define SOLVERS_COST_H

#include <casadi/casadi.hpp>

#include "symbolic/expression.h"
#include "utils/eigen_wrapper.h"

namespace damotion {
namespace optimisation {

class Cost {
   public:
    Cost() = default;
    ~Cost() = default;

    Cost(const symbolic::Expression &expr, bool grad = false, bool hes = false);

    void SetObjectiveFunction(const casadi::Function &f) { obj_ = f; }
    void SetGradientFunction(const casadi::Function &f) { grad_ = f; }
    void SetHessianFunction(const casadi::Function &f) { hes_ = f; }

    utils::casadi::FunctionWrapper &ObjectiveFunction() { return obj_; }
    utils::casadi::FunctionWrapper &GradientFunction() { return grad_; }
    utils::casadi::FunctionWrapper &HessianFunction() { return hes_; }

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
    // Cost weighting
    double w_;

    /**
     * @brief Objective function
     *
     */
    utils::casadi::FunctionWrapper obj_;

    /**
     * @brief Gradient function
     *
     */
    utils::casadi::FunctionWrapper grad_;

    /**
     * @brief Hessian function
     *
     */
    utils::casadi::FunctionWrapper hes_;

    // Number of variable inputs
    int nx_ = 0;
    // Number of parameter inputs
    int np_ = 0;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_COST_H */
