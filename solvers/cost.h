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

    Cost(const symbolic::Expression &ex, const std::string &name = "",
         bool grd = false, bool hes = false);

    utils::casadi::FunctionWrapper &ObjectiveFunction() { return obj_; }
    utils::casadi::FunctionWrapper &GradientFunction() { return grad_; }
    utils::casadi::FunctionWrapper &HessianFunction() { return hes_; }

    /**
     * @brief Whether the constraint has a Gradient
     *
     * @return true
     * @return false
     */
    const bool HasGradient() const { return has_grd_; }

    /**
     * @brief Whether the constraint has a Hessian
     *
     * @return true
     * @return false
     */
    const bool HasHessian() const { return has_hes_; }

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

    bool has_grd_ = false;
    bool has_hes_ = false;

    // Number of variable inputs
    int nx_ = 0;
    // Number of parameter inputs
    int np_ = 0;

    // Name of the cost
    std::string name_;

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

    void SetObjectiveFunction(const casadi::Function &f) { obj_ = f; }
    void SetGradientFunction(const casadi::Function &f) {
        grad_ = f;
        has_grd_ = true;
    }
    void SetHessianFunction(const casadi::Function &f) {
        hes_ = f;
        has_hes_ = true;
    }

    /**
     * @brief Creates a unique id for each cost
     *
     * @return int
     */
    int CreateID() {
        static int next_id = 0;
        int id = next_id;
        next_id++;
        return id;
    }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_COST_H */
