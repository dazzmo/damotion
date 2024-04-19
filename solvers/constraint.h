#ifndef SOLVERS_CONSTRAINT_H
#define SOLVERS_CONSTRAINT_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>

#include "solvers/bounds.h"
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
            std::make_shared<utils::casadi::VectorFunctionWrapper>(
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
        UpdateBounds(bounds);
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

    std::shared_ptr<common::VectorFunction> &ConstraintFunction() {
        return con_;
    }
    std::shared_ptr<common::FunctionBase<MatrixType>> &JacobianFunction() {
        return jac_;
    }
    std::shared_ptr<common::FunctionBase<MatrixType>> &HessianFunction() {
        return hes_;
    }

    /**
     * @brief Flag to indicate if the constraint has a Jacobian
     *
     * @return true
     * @return false
     */
    const bool HasJacobian() const { return has_jac_; }

    /**
     * @brief Flag to indicate whether the constraint has a Hessian
     *
     * @return true
     * @return false
     */
    const bool HasHessian() const { return has_hes_; }

    void SetBoundsType(const BoundsType &type) {
        bounds_type_ = type;
        SetBounds(ub_, lb_, bounds_type_);
    }

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
     * @brief Sets constraint bounds to a custom interval
     *
     * @param lb
     * @param ub
     */
    void UpdateBounds(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub) {
        bounds_type_ = BoundsType::kCustom;
        lb_ = lb;
        ub_ = ub;

        // Indicate constraint was updated
        IsUpdated() = true;
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
            c_norm = con_->getOutput(0).lpNorm<1>();
        } else if (p == 2) {
            c_norm = con_->getOutput(0).lpNorm<2>();
        } else if (p == Eigen::Infinity) {
            c_norm = con_->getOutput(0).lpNorm<Eigen::Infinity>();
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

    // bool IsSparse();

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
        const std::shared_ptr<common::VectorFunction> &f) {
        con_ = f;
    }

    /**
     * @brief Set the Jacobian Function object
     *
     * @param f
     */
    void SetJacobianFunction(
        const std::shared_ptr<common::FunctionBase<MatrixType>> &f) {
        jac_ = f;
        has_jac_ = true;
    }

    /**
     * @brief Set the Hessian Function object
     *
     * @param f
     */
    void SetHessianFunction(
        const std::shared_ptr<common::FunctionBase<MatrixType>> &f) {
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

    // Constraint
    std::shared_ptr<common::VectorFunction> con_;
    // Jacobian
    std::shared_ptr<common::FunctionBase<MatrixType>> jac_;
    // Hessian of vector-product
    std::shared_ptr<common::FunctionBase<MatrixType>> hes_;

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

template <typename MatrixType>
class LinearConstraintBase : public ConstraintBase<MatrixType> {
   public:
    /**
     * @brief Construct a new Linear Constraint object of the form \f$ A x + b
     * \f$
     *
     * @param name Name of the constraint. Default name given if provided ""
     * @param A Vector of coefficient matrices
     * @param b
     * @param p Parameters that A and b depend on
     * @param bounds
     * @param jac
     */
    LinearConstraintBase(const std::string &name, const casadi::SX &A,
                         const casadi::SX &b, const casadi::SXVector &p,
                         const BoundsType &bounds, bool jac = true)
        : Constraint(name, "linear_constraint") {
        ConstructConstraint(A, b, p, bounds, jac);
    }

    LinearConstraintBase(const std::string &name, const Eigen::MatrixXd &A,
                         const Eigen::VectorXd &b, const BoundsType &bounds,
                         bool jac = true)
        : Constraint(name, "linear_constraint") {
        // Constant vector b
        casadi::DM Ad, bd;
        damotion::utils::casadi::toCasadi(b, bd);
        damotion::utils::casadi::toCasadi(A, Ad);
        casadi::SX bsx = bd;
        casadi::SX Asx = Ad;

        // Construct constraint
        ConstructConstraint(Asx, bsx, {}, bounds, jac);
    }

    LinearConstraintBase(const std::string &name, const sym::Expression &ex,
                         const BoundsType &bounds, bool jac = true)
        : Constraint(name, "linear_constraint") {
        // Extract linear form
        casadi::SX A, b;
        casadi::SX::linear_coeff(ex, ex.Variables()[0], A, b, true);

        ConstructConstraint(A, b, ex.Parameters(), bounds, jac);
    }

    void SetCallback(
        const typename common::CallbackFunction<MatrixType>::f_callback_ &fA,
        const typename common::CallbackFunction<Eigen::VectorXd>::f_callback_
            &fb) {
        fA_ = std::make_shared<common::CallbackFunction<MatrixType>>(1, 1, fA);
        fb_ = std::make_shared<common::CallbackFunction<Eigen::VectorXd>>(1, 1,
                                                                          fb);
    }

    /**
     * @brief The coefficient matrix A for the expression A x + b.
     *
     * @return const Eigen::MatrixXd&
     */
    const Eigen::Ref<const Eigen::MatrixXd> A() { return fA_->getOutput(0); }

    /**
     * @brief The constant vector for the linear constraint A x + b
     *
     * @return const Eigen::VectorXd&
     */
    const Eigen::Ref<const Eigen::VectorXd> b() { return fb_->getOutput(0); }

   private:
    std::shared_ptr<common::FunctionBase<MatrixType>> fA_;
    std::shared_ptr<common::VectorFunction> fb_;

    /**
     * @brief Compute the constraint with use of A and b
     *
     * @param input
     * @param out
     */
    void ConstraintCallback(const common::InputRefVector &input,
                            std::vector<Eigen::VectorXd> &out) {
        out[0] = fA_->getOutput(0) * input[0] + fb_->getOutput(0);
    }

    /**
     * @brief Compute the Jacobian of the constraint with A
     *
     * @param input
     * @param out
     */
    void JacobianCallback(const common::InputRefVector &input,
                          std::vector<MatrixType> &out) {
        out[0] = fA_->getOutput(0);
    }

    void ConstructConstraint(const casadi::SX &A, const casadi::SX &b,
                             const casadi::SXVector &p,
                             const BoundsType &bounds, bool jac = true,
                             bool sparse = false) {
        casadi::SXVector in = {};
        int nvar = 0;
        assert(A.rows() == b.rows() && "A and b must be same dimension!");

        // Create constraint dimensions and update bounds
        this->Resize(b.rows(), in.size(), p.size());
        this->UpdateBounds(bounds);

        // Add any parameters that define A and b
        for (const casadi::SX &pi : p) {
            in.push_back(pi);
        }

        if (std::is_same<MatrixType, Eigen::SparseMatrix<double>>::value) {
            fA_ = std::make_shared<utils::casadi::FunctionWrapper<MatrixType>>(
                casadi::Function(this->name() + "_A", in, {A}));
        } else {
            fA_ = std::make_shared<utils::casadi::FunctionWrapper<MatrixType>>(
                casadi::Function(this->name() + "_A", in, {densify(A)}));
        }

        fb_ = std::make_shared<utils::casadi::FunctionWrapper<Eigen::VectorXd>>(
            casadi::Function(this->name() + "_b", in, {densify(b)}));

        std::shared_ptr<common::CallbackFunction<Eigen::VectorXd>> con_cb =
            std::make_shared<common::CallbackFunction<Eigen::VectorXd>>(
                in.size(), 1,
                [this](const common::InputRefVector &in,
                       std::vector<Eigen::VectorXd> &out) {
                    this->ConstraintCallback(in, out);
                });
        std::shared_ptr<common::CallbackFunction<MatrixType>> jac_cb =
            std::make_shared<common::CallbackFunction<MatrixType>>(
                in.size(), 1,
                [this](const common::InputRefVector &in,
                       std::vector<MatrixType> &out) {
                    this->JacobianCallback(in, out);
                });

        // Set output sizes for the callback
        con_cb->InitOutput(0, Eigen::VectorXd::Zero(A.size1()));
        // Construct jacobian
        if (std::is_same<MatrixType, Eigen::SparseMatrix<double>>::value) {
            std::vector<casadi_int> rows, cols;
            A.sparsity().get_triplet(rows, cols);
            jac_cb->InitOutput(0, utils::casadi::CreateSparseEigenMatrix(
                                      A.sparsity(), rows, cols));
        } else {
            jac_cb->InitOutput(0, Eigen::MatrixXd::Zero(A.size1(), A.size2()));
        }

        // Create functions through callbacks
        this->SetConstraintFunction(con_cb);
        this->SetJacobianFunction(jac_cb);
    }
};

typedef LinearConstraintBase<Eigen::MatrixXd> LinearConstraint;
typedef LinearConstraintBase<Eigen::SparseMatrix<double>>
    SparseLinearConstraint;

template <typename MatrixType>
class BoundingBoxConstraintBase : public ConstraintBase<MatrixType> {
   public:
    BoundingBoxConstraintBase(const std::string &name,
                              const Eigen::VectorXd &lb,
                              const Eigen::VectorXd &ub)
        : ConstraintBase<MatrixType>(name, "bounding_box_constraint") {
        assert(lb.rows() == ub.rows() && "lb and ub must be same dimension!");

        int n = lb.rows();
        // Resize the constraint
        this->Resize(n, n, 0);
        // Update bounds
        this->UpdateBounds(lb, ub);

        casadi::SX x = casadi::SX::sym("x", n);

        casadi::Function f = casadi::Function(this->name(), {x}, {densify(x)});
        casadi::Function fjac = casadi::Function(this->name() + "_jac", {x},
                                                 {densify(jacobian(x, x))});

        this->SetConstraintFunction(
            std::make_shared<utils::casadi::FunctionWrapper<Eigen::VectorXd>>(
                f));
        this->SetJacobianFunction(
            std::make_shared<utils::casadi::FunctionWrapper<MatrixType>>(fjac));
    }

   private:
};

typedef BoundingBoxConstraintBase<Eigen::MatrixXd> BoundingBoxConstraint;
typedef BoundingBoxConstraintBase<Eigen::SparseMatrix<double>>
    SparseBoundingBoxConstraint;

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_CONSTRAINT_H */
