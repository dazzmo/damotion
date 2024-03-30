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
     * @param c The expression to compute the constraint and derivatives from
     * @param bounds Bounds for the constraint (e.g. equality, positive,
     * negative ...)
     * @param name Optional name for the constraint, defaults to "c_" where _
     * is the id of the constraint
     * @param jac Flag to compute the jacobian of c with respect to each input
     * variable
     * @param hes Flag to compute the hessian of c with respect to each input
     * variable
     */
    Constraint(const symbolic::Expression &c, const BoundsType &bounds,
               const std::string &name = "", bool jac = true, bool hes = true);

    /**
     * @brief Construct a new Constraint object without an expression
     *
     * @param name Optional name for the constraint, defaults to "c_" where _
     * is the id of the constraint
     */
    Constraint(const std::string &name = "");

    utils::casadi::FunctionWrapper &ConstraintFunction() { return con_; }
    utils::casadi::FunctionWrapper &JacobianFunction() { return jac_; }
    utils::casadi::FunctionWrapper &HessianFunction() { return hes_; }

    void SetBoundsType(const BoundsType &type) {
        bounds_type_ = type;
        SetBounds(ub_, lb_, bounds_type_);
    }

    /**
     * @brief Whether the constraint has a Jacobian
     *
     * @return true
     * @return false
     */
    const bool HasJacobian() const { return has_jac_; }

    /**
     * @brief Whether the constraint has a Hessian
     *
     * @return true
     * @return false
     */
    const bool HasHessian() const { return has_hes_; }

    /**
     * @brief Name of the constraint
     *
     * @return const std::string
     */
    const std::string name() const { return name_; }

    /**
     * @brief Dimension of the constraint
     *
     * @return const int
     */
    const int Dimension() const { return dim_; }

    /**
     * @brief The current type of bounds for the constraint
     *
     * @return const BoundsType&
     */
    const BoundsType &GetBoundsType() const { return bounds_type_; }

    /**
     * @brief Updates the bounds for the constraint according to type
     *
     * @param type
     */
    void UpdateBounds(const BoundsType &type) {
        bounds_type_ = type;
        SetBounds(ub_, lb_, bounds_type_);
    }

    /**
     * @brief Sets constaint bounds to a custom interval
     *
     * @param lb
     * @param ub
     */
    void UpdateBounds(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub) {
        bounds_type_ = BoundsType::kCustom;
        lb_ = lb;
        ub_ = ub;
    }

    /**
     * @brief Constraint lower bound (dim x 1)
     *
     * @return const Eigen::VectorXd&
     */
    const Eigen::VectorXd &LowerBound() const { return lb_; }
    Eigen::VectorXd &LowerBound() { return lb_; }

    /**
     * @brief Constraint upper bound (dim x 1)
     *
     * @return const Eigen::VectorXd&
     */
    const Eigen::VectorXd &UpperBound() const { return ub_; }
    Eigen::VectorXd &UpperBound() { return ub_; }

    const int &NumberOfInputVariables() const { return nx_; }
    const int &NumberOfInputParameters() const { return np_; }

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
    /**
     * @brief Resizes the constraint dimensions.
     *
     * @param dim Dimension of the constraint
     * @param nx Number of input variables
     * @param np Number of input parameters
     */
    void Resize(int dim, int nx, int np) {
        dim_ = dim;
        nx_ = nx;
        np_ = np;

        double inf = std::numeric_limits<double>::infinity();
        ub_ = inf * Eigen::VectorXd::Ones(dim_);
        lb_ = -inf * Eigen::VectorXd::Ones(dim_);
    }

    /**
     * @brief Set the Constraint Function object
     *
     * @param f
     */
    void SetConstraintFunction(const casadi::Function &f) { con_ = f; }

    /**
     * @brief Set the Jacobian Function object
     *
     * @param f
     */
    void SetJacobianFunction(const casadi::Function &f) {
        jac_ = f;
        has_jac_ = true;
    }

    /**
     * @brief Set the Hessian Function object
     *
     * @param f
     */
    void SetHessianFunction(const casadi::Function &f) {
        hes_ = f;
        has_hes_ = true;
    }

   private:
    // Dimension of the constraint
    int dim_ = 0;

    // Name of the constraint
    std::string name_;

    // Flags to indicate if constraint can compute derivatives
    bool has_jac_ = false;
    bool has_hes_ = false;

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

    /**
     * @brief Creates a unique id for each constraint
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

class LinearConstraint : public Constraint {
   public:
    LinearConstraint(const std::vector<Eigen::MatrixXd> &A,
                     const Eigen::VectorXd &b, const BoundsType &bounds,
                     const std::string &name = "", bool jac = true)
        : Constraint(name) {
        casadi::SXVector in = {}, jacobians = {};
        int nvar = 0;
        // Constant vector b
        casadi::DM bdm;
        damotion::utils::casadi::toCasadi(b, bdm);
        casadi::SX bsx = bdm;
        casadi::SX linear_constraint = bsx;
        for (const Eigen::MatrixXd &Ai : A) {
            assert(Ai.rows() == b.rows() && "A and b must be same dimension!");

            // Create constraints
            casadi::DM Adm;
            damotion::utils::casadi::toCasadi(Ai, Adm);
            casadi::SX Asx = Adm;

            casadi::SX xi = casadi::SX::sym("x", Ai.cols());
            linear_constraint += mtimes(Asx, xi);
            jacobians.push_back(Asx);
            in.push_back(xi);
            nvar += 1;
        }
        // Resize the constraint
        Resize(b.rows(), nvar, 0);
        // Update bounds
        UpdateBounds(bounds);

        // Create the constraint
        casadi::Function f =
            casadi::Function(this->name(), in, {linear_constraint, bsx});
        casadi::Function fjac =
            casadi::Function(this->name() + "_jac", in, jacobians);

        SetConstraintFunction(f);
        SetJacobianFunction(fjac);
    }

    /**
     * @brief Construct a new Linear Constraint object of the form \f$ A x + b
     * \f$
     *
     * @param A
     * @param b
     * @param p Parameters that A and b depend on
     * @param bounds
     * @param name
     * @param jac
     */
    LinearConstraint(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
                     const BoundsType &bounds, const std::string &name = "",
                     bool jac = true)
        : LinearConstraint(std::vector<Eigen::MatrixXd>({A}), b, bounds, name,
                           jac) {}

    /**
     * @brief The i-th coefficient matrix for the linear constraint A_1 x_1 +
     * A_2 x_2 + ... A_n x_n + b
     *
     * @return const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd &A(const int &i) {
        return JacobianFunction().getOutput(i);
    }

    /**
     * @brief Construct a new Linear Constraint object of the form \f$ A_0 x_0 +
     * A_1 x_1 + \hdots + A_n x_n + b \f$
     *
     * @param A Vector of coefficient matrices
     * @param b
     * @param p Parameters that A and b depend on
     * @param bounds
     * @param name
     * @param jac
     */
    LinearConstraint(const casadi::SXVector &A, const casadi::SX &b,
                     const casadi::SXVector &p, const BoundsType &bounds,
                     const std::string &name = "", bool jac = true)
        : Constraint(name) {
        casadi::SXVector in = {};
        int nvar = 0;
        casadi::SX linear_constraint = b;
        // Add each coefficient matrix
        for (casadi::SX Ai : A) {
            assert(Ai.rows() == b.rows() && "A and b must be same dimension!");
            // Create optimisation variables for each coefficient matrix
            casadi::SX xi = casadi::SX::sym("x", Ai.columns());
            in.push_back(xi);
            // Add linear expression A_i x_i
            linear_constraint += casadi::SX::mtimes(Ai, xi);
        }
        // Create constraint dimensions and update bounds
        Resize(b.rows(), in.size(), p.size());
        UpdateBounds(bounds);

        // Add any parameters that define A and b
        for (const casadi::SX &pi : p) {
            in.push_back(pi);
        }

        // Create the constraint, including the constant term b as well
        casadi::Function f = casadi::Function(name, in, {linear_constraint, b});
        casadi::Function fjac = casadi::Function(name + "_jac", in, A);
        SetConstraintFunction(f);
        SetJacobianFunction(fjac);
    }

    /**
     * @brief Construct a new Linear Constraint object of the form \f$ A x + b
     * \f$
     *
     * @param A
     * @param b
     * @param p Parameters that A and b depend on
     * @param bounds
     * @param name
     * @param jac
     */
    LinearConstraint(const casadi::SX &A, const casadi::SX &b,
                     const casadi::SXVector &p, const BoundsType &bounds,
                     const std::string &name = "", bool jac = true)
        : LinearConstraint(casadi::SXVector({A}), b, p, bounds, name, jac) {}


    /**
     * @brief The coefficient matrix A for the expression A x + b.
     *
     * @return const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd &A() { return A(0); }

    /**
     * @brief The constant vector for the linear constraint A x + b
     *
     * @return const Eigen::VectorXd&
     */
    const Eigen::MatrixXd &b() {
        return ConstraintFunction().getOutput(NumberOfInputVariables());
    }

   private:
    Eigen::MatrixXd A_;
    Eigen::VectorXd b_;
};

class BoundingBoxConstraint : public Constraint {
   public:
    BoundingBoxConstraint(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
                          const std::string &name = "")
        : Constraint(name) {
        assert(lb.rows() == ub.rows() && "lb and ub must be same dimension!");

        int n = lb.rows();
        // Resize the constraint
        Resize(n, n, 0);
        // Update bounds
        UpdateBounds(lb, ub);

        casadi::SX x = casadi::SX::sym("x", n);

        casadi::Function f = casadi::Function(this->name(), {x}, {x});
        casadi::Function fjac =
            casadi::Function(this->name() + "_jac", {x}, {jacobian(x, x)});

        SetConstraintFunction(f);
        SetJacobianFunction(fjac);
    }

   private:
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_CONSTRAINT_H */
