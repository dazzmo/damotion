#ifndef SOLVERS_CONSTRAINT_H
#define SOLVERS_CONSTRAINT_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>

#include "solvers/bounds.h"
#include "utils/eigen_wrapper.h"

namespace damotion {
namespace optimisation {

class Constraint {
   public:
    Constraint() = default;
    ~Constraint() = default;

    Constraint(const std::string &name, const int dim)
        : name_(name), dim_(dim) {
        lb_ = -Eigen::VectorXd::Ones(dim_);
        ub_ = Eigen::VectorXd::Ones(dim_);
    }

    void SetSymbolicConstraint(const casadi::SX &c) { c_ = c; }
    casadi::SX &SymbolicConstraint() { return c_; }

    void SetSymbolicInputs(const casadi::SXVector &inputs) { inputs_ = inputs; }
    casadi::SXVector &SymbolicInputs() { return inputs_; }

    void SetConstraintFunction(casadi::Function &f) { con_ = f; }
    void SetJacobianFunction(casadi::Function &f) { jac_ = f; }
    void SetHessianFunction(casadi::Function &f) { hes_ = f; }
    void SetLinearisedConstraintFunction(casadi::Function &f) { lin_ = f; }

    utils::casadi::FunctionWrapper &ConstraintFunction() { return con_; }
    utils::casadi::FunctionWrapper &JacobianFunction() { return jac_; }
    utils::casadi::FunctionWrapper &HessianFunction() { return hes_; }
    utils::casadi::FunctionWrapper &LinearisedConstraintFunction() { return lin_; }

    const BoundsType &GetBoundsType() const { return bounds_type_; }
    void SetBoundsType(const BoundsType &type) {
        bounds_type_ = type;
        SetBounds(ub_, lb_, bounds_type_);
    }

    /**
     * @brief Name of the constraint
     *
     * @return const std::string&
     */
    const std::string &name() const { return name_; }

    /**
     * @brief Dimension of the constraint
     *
     * @return const int
     */
    const int dim() const { return dim_; }

    /**
     * @brief The index of the constraint within the constraint vector
     *
     * @return const int&
     */
    const int &idx() const { return idx_; }

    /**
     * @brief Set the index of this constraint within the program constraint
     * vector
     *
     * @param idx
     */
    void SetIndex(const int idx) { idx_ = idx; }

    /**
     * @brief Constraint lower bound (dim x 1)
     *
     * @return const Eigen::VectorXd&
     */
    const Eigen::VectorXd &lb() const { return lb_; }
    Eigen::VectorXd &lb() { return lb_; }

    /**
     * @brief Constraint upper bound (dim x 1)
     *
     * @return const Eigen::VectorXd&
     */
    const Eigen::VectorXd &ub() const { return ub_; }
    Eigen::VectorXd &ub() { return ub_; }

    /**
     * @brief Tests whether the p-norm of the constraint is within
     * the threshold eps.
     *
     * @param p The norm of the constraint (use Eigen::Infinity for the
     * infinity norm)
     * @param eps
     * @return true
     * @return false
     */
    bool CheckViolation(const int &p = 2, const double &eps = 1e-6) {
        // Determine if constraint within threshold
        double c_norm = 0.0;
        if (p == 1) {
            c_norm = con_.getOutput(0).lpNorm<1>();
        } else if (p == 2) {
            c_norm = con_.getOutput(0).lpNorm<2>();
        } else if (p == Eigen::Infinity) {
            c_norm = con_.getOutput(0).lpNorm<Eigen::Infinity>();
        }

        return c_norm <= eps;
    }

   private:
    // Dimension of the constraint
    int dim_ = 0;
    // Starting index of the constraint within the overall constraint vector
    int idx_ = 0;

    // Name of the constraint
    std::string name_;

    BoundsType bounds_type_ = BoundsType::kUnbounded;

    // Underlying symbolic representation of constraint
    casadi::SX c_;
    // Symbolic input vector
    casadi::SXVector inputs_;

    // Constraint lower bound
    Eigen::VectorXd lb_;
    // Constraint upper bound
    Eigen::VectorXd ub_;

    // Constraint
    utils::casadi::FunctionWrapper con_;
    // Jacobian
    utils::casadi::FunctionWrapper jac_;
    // Hessian of vector-product
    utils::casadi::FunctionWrapper hes_;

    // Linearised constraint
    utils::casadi::FunctionWrapper lin_;

};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_CONSTRAINT_H */
