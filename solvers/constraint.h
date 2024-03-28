#ifndef SOLVERS_CONSTRAINT_H
#define SOLVERS_CONSTRAINT_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>

#include "solvers/bounds.h"
#include "symbolic/expression.h"
#include "utils/eigen_wrapper.h"

namespace damotion {
namespace optimisation {

class Constraint {
   public:
    Constraint() = default;
    ~Constraint() = default;

    /**
     * @brief Construct a new Constraint object from the symbolic expression c
     *
     * @param c
     * @param jac
     * @param hes
     */
    Constraint(const symbolic::Expression &c,
               const BoundsType &bounds = BoundsType::kUnbounded,
               bool jac = false, bool hes = false);

    void SetConstraintFunction(const casadi::Function &f) { con_ = f; }
    void SetJacobianFunction(const casadi::Function &f) { jac_ = f; }
    void SetHessianFunction(const casadi::Function &f) { hes_ = f; }

    utils::casadi::FunctionWrapper &ConstraintFunction() { return con_; }
    utils::casadi::FunctionWrapper &JacobianFunction() { return jac_; }
    utils::casadi::FunctionWrapper &HessianFunction() { return hes_; }

    const BoundsType &GetBoundsType() const { return bounds_type_; }
    void SetBoundsType(const BoundsType &type) {
        bounds_type_ = type;
        SetBounds(ub_, lb_, bounds_type_);
    }

    /**
     * @brief Dimension of the constraint
     *
     * @return const int
     */
    const int dim() const { return dim_; }

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

   protected:
    void SetDimension(const int &dim) { dim_ = dim; }

   private:
    // Dimension of the constraint
    int dim_ = 0;

    BoundsType bounds_type_ = BoundsType::kUnbounded;

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

    // Number of variable inputs
    int nx_ = 0;
    // Number of parameter inputs
    int np_ = 0;
};

class LinearConstraint : public Constraint {
   public:
    LinearConstraint(const Eigen::MatrixXd &A, const Eigen::VectorXd &b) {
        // Create constraints
        casadi::DM Adm, bdm;
        damotion::utils::casadi::toCasadi(A, Adm);
        damotion::utils::casadi::toCasadi(b, bdm);

        casadi::SX Asx = Adm, bsx = bdm;

        // Create constraint, including the constant term b as well
        casadi::SX x = casadi::SX::sym("x", A.cols());
        casadi::SX c = casadi::SX::mtimes(Asx, x) + bsx;
        // Create the constraint
        casadi::Function f = casadi::Function("lin_con", {x}, {c, bsx});
        casadi::Function fjac = casadi::Function("lin_con_jac", {x}, {Asx});
        SetConstraintFunction(f);
        SetJacobianFunction(fjac);

        SetDimension(A.rows());
    }

    LinearConstraint(const casadi::SX &A, const casadi::SX &b,
                     const casadi::SXVector &p) {
        // Create constraint
        casadi::SX x = casadi::SX::sym("x", A.size2());

        casadi::SXVector in = {x};
        // Add any parameters that define A and b
        for (const casadi::SX &pi : p) {
            in.push_back(pi);
        }

        casadi::SX linear_constraint = casadi::SX::mtimes(A, x) + b;

        // Create the constraint, including the constant term b as well
        casadi::Function f =
            casadi::Function("lin_con", in, {linear_constraint, b});
        casadi::Function fjac = casadi::Function("lin_con_jac", in, {A});
        SetConstraintFunction(f);
        SetJacobianFunction(fjac);

        SetDimension(A.rows());
    }

    const Eigen::MatrixXd &A() { return JacobianFunction().getOutput(0); }
    const Eigen::VectorXd &b() { return ConstraintFunction().getOutput(1); }

   private:
    Eigen::MatrixXd A_;
    Eigen::VectorXd b_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_CONSTRAINT_H */
