#ifndef CONSTRAINTS_BASE_H
#define CONSTRAINTS_BASE_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>

#include "optimisation/bounds.h"
#include "symbolic/expression.h"
#include "utils/eigen_wrapper.h"

namespace damotion {
namespace optimisation {

namespace sym = damotion::symbolic;

template <typename MatrixType>
class ConstraintBase {
   public:
    ConstraintBase() = default;
    ~ConstraintBase() = default;

    /**
     * @brief Construct a new ConstraintBase object without an expression
     *
     * @param name Name for the constraint, if given "", provides a default name
     * based on the constraint id
     * @param constraint_type Constraint type to use for the name if default
     * names are chosen
     */
    ConstraintBase(const std::string &name,
                   const std::string &constraint_type) {
        // Set default name for constraint
        if (name != "") {
            name_ = name;
        } else {
            name_ = constraint_type + "_" + std::to_string(CreateID());
        }
    }

    /**
     * @brief Construct a new ConstraintBase object from the symbolic
     * expression c
     *
     * @param name Optional name for the constraint, if set to "", creates a
     * default name for the constraint base on the id of the constraint
     * @param c The expression to compute the constraint and derivatives from
     * @param bounds Bounds for the constraint (e.g. equality, positive,
     * negative ...)
     * @param jac Flag to compute the jacobian of c with respect to each input
     * variable
     * @param hes Flag to compute the hessian of c with respect to each input
     * variable
     */
    ConstraintBase(const std::string &name, const symbolic::Expression &c,
                   const BoundsType &bounds, bool jac, bool hes)
        : ConstraintBase(name, "constraint") {
        // Resize the constraint
        Resize(c.size1(), c.Variables().size(), c.Parameters().size());

        // Create functions to compute the constraint and derivatives given the
        // variables and parameters
        casadi::SXVector in = c.Variables();
        for (const casadi::SX &pi : c.Parameters()) {
            in.push_back(pi);
        }

        // Constraint
        SetConstraintFunction(
            std::make_shared<utils::casadi::FunctionWrapper<Eigen::VectorXd>>(
                casadi::Function(this->name(), in, {c})));

        // Jacobian
        if (jac) {
            casadi::SXVector jacobians;
            for (const casadi::SX &xi : c.Variables()) {
                jacobians.push_back(jacobian(c, xi));
            }
            // Wrap the functions
            SetJacobianFunction(
                std::make_shared<utils::casadi::FunctionWrapper<MatrixType>>(
                    casadi::Function(this->name() + "_jac", in, jacobians)));
        }

        if (hes) {
            // TODO - Hessians
        }

        // Update bounds for the constraint
        SetBounds(bounds);
    }

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
     * @brief Evaluate the constraint with the current input variables and
     * parameters, indicating if jacobian and hessians are required
     *
     * @param x Variables for the constraint
     * @param p Parameters for the constraint
     * @param jac Whether to also compute the Jacobian
     */
    virtual void eval(const common::InputRefVector &x,
                      const common::InputRefVector &p, bool jac = true) const {
        VLOG(10) << this->name() << " eval()";
        common::InputRefVector in = {};
        for (int i = 0; i < x.size(); ++i) in.push_back(x[i]);
        for (int i = 0; i < p.size(); ++i) in.push_back(p[i]);

        // Call necessary constraint functions
        this->con_->call(in);
        if (jac) {
            this->jac_->call(in);
        }
    }

    /**
     * @brief Evaluate the dual-variable-Hessian product via a given strategy
     *
     * @param x
     * @param p
     */
    void eval_hessian(const common::InputRefVector &x,
                      const common::InputRefVector &l,
                      const common::InputRefVector &p) {
        // Create input for the lambda-hessian product
        common::InputRefVector in = {};
        for (int i = 0; i < x.size(); ++i) in.push_back(x[i]);
        for (int i = 0; i < p.size(); ++i) in.push_back(p[i]);
        for (int i = 0; i < l.size(); ++i) in.push_back(l[i]);

        // Call necessary constraint functions
        this->hes_->call(in);
    }

    /**
     * @brief Returns the most recent evaluation of the constraint
     *
     * @return const Eigen::VectorXd&
     */
    virtual const Eigen::VectorXd &Vector() const {
        VLOG(10) << name_ << " Vector = " << con_->getOutput(0);
        return con_->getOutput(0);
    }
    /**
     * @brief The Jacobian of the constraint with respect to the i-th variable
     * vector
     *
     * @param i
     * @return const MatrixType&
     */
    virtual const MatrixType &Jacobian(const int &i) const {
        VLOG(10) << name_ << " Jacobian " << i << " = " << jac_->getOutput(i);
        return jac_->getOutput(i);
    }
    /**
     * @brief Returns the Hessian block with respect to the variables xi and xj.
     * Please note that this formulation produces only the lower-triangular
     * component of the Hessian, so i >= j.
     *
     * @param i
     * @param j
     * @return const MatrixType&
     */
    const MatrixType &Hessian(const int &i, const int &j) const {
        // Determine the hessian block index
        int idx = 0;
        // TODO
        return hes_->getOutput(idx);
    }

    /**
     * @brief Set the Bounds type for the constraint.
     *
     * @param type
     */
    void SetBounds(const BoundsType &type) {
        bounds_type_ = type;
        SetBoundsByType(ub_, lb_, bounds_type_);
    }

    /**
     * @brief Sets constraint bounds to a custom interval
     *
     * @param lb
     * @param ub
     */
    void SetBounds(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub) {
        bounds_type_ = BoundsType::kCustom;
        lb_ = lb;
        ub_ = ub;

        // Indicate constraint was updated
        IsUpdated() = true;
    }

    /**
     * @brief The current type of bounds for the constraint
     *
     * @return const BoundsType&
     */
    const BoundsType &GetBoundsType() const { return bounds_type_; }

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

    /**
     * @brief Number of input variable vectors used to determine the constraint
     *
     * @return const int&
     */
    const int &NumberOfInputVariables() const { return nx_; }

    /**
     * @brief Number of parameters used to determine the constraint
     *
     * @return const int&
     */
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
            c_norm = this->Vector().template lpNorm<1>();
        } else if (p == 2) {
            c_norm = this->Vector().template lpNorm<2>();
        } else if (p == Eigen::Infinity) {
            c_norm = this->Vector().template lpNorm<Eigen::Infinity>();
        }

        return c_norm <= eps;
    }

    /**
     * @brief Indicates if the constraint has been updated since it was last
     * used, can be set to true and false.
     *
     * @return true
     * @return false
     */
    const bool &IsUpdated() const { return updated_; }
    bool &IsUpdated() { return updated_; }

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
    void SetConstraintFunction(
        const std::shared_ptr<common::Function<Eigen::VectorXd>> &f) {
        con_ = f;
    }

    /**
     * @brief Set the Jacobian Function object
     *
     * @param f
     */
    void SetJacobianFunction(
        const std::shared_ptr<common::Function<MatrixType>> &f) {
        jac_ = f;
        has_jac_ = true;
    }

    /**
     * @brief Set the Hessian Function object
     *
     * @param f
     */
    void SetHessianFunction(
        const std::shared_ptr<common::Function<MatrixType>> &f) {
        hes_ = f;
        has_hes_ = true;
    }

   private:
    // Dimension of the constraint
    int dim_ = 0;

    // Flag to indicate if the constraint has changed since it was used
    bool updated_;

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

    // Constraint function pointer
    std::shared_ptr<common::Function<Eigen::VectorXd>> con_;
    // Jacobian function pointer
    std::shared_ptr<common::Function<MatrixType>> jac_;
    // Hessian of vector-product function pointer
    std::shared_ptr<common::Function<MatrixType>> hes_;

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

typedef ConstraintBase<Eigen::MatrixXd> Constraint;
typedef ConstraintBase<Eigen::SparseMatrix<double>> SparseConstraint;

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_BASE_H */
