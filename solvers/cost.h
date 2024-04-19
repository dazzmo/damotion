#ifndef SOLVERS_COST_H
#define SOLVERS_COST_H

#include <casadi/casadi.hpp>

#include "symbolic/expression.h"
#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"

namespace damotion {
namespace optimisation {

namespace sym = damotion::symbolic;

template <typename MatrixType>
class CostBase {
   public:
    CostBase() = default;
    ~CostBase() = default;

    CostBase(const std::string &name, const std::string &cost_type) {
        // Set default name for constraint
        if (name != "") {
            name_ = name;
        } else {
            name_ = cost_type + "_" + std::to_string(CreateID());
        }
    }

    CostBase(const std::string &name, const symbolic::Expression &ex,
             bool grd = false, bool hes = false)
        : CostBase(name, "cost") {
        // Get input sizes
        nx_ = ex.Variables().size();
        np_ = ex.Parameters().size();

        // Create functions to compute the constraint and derivatives given the
        // variables and parameters

        // Input vectors {x, p}
        casadi::SXVector in = ex.Variables();
        for (const casadi::SX &pi : ex.Parameters()) {
            in.push_back(pi);
        }

        // Create functions for each and wrap them
        // Constraint
        SetObjectiveFunction(
            std::make_shared<utils::casadi::ScalarFunctionWrapper>(
                casadi::Function(name, in, {ex})));
        // Jacobian
        if (grd) {
            casadi::SXVector gradients;
            for (const casadi::SX &xi : ex.Variables()) {
                gradients.push_back(gradient(ex, xi));
            }
            // Wrap the functions
            SetGradientFunction(
                std::make_shared<utils::casadi::VectorFunctionWrapper>(
                    casadi::Function(name + "_grd", in, gradients)));
        }

        // Hessians
        if (hes) {
            casadi::SXVector hessians;
            // For each combination of input variables, compute the hessians
            for (int i = 0; i < ex.Variables().size(); ++i) {
                casadi::SX xi = ex.Variables()[i];
                for (int j = i; j < ex.Variables().size(); ++j) {
                    casadi::SX xj = ex.Variables()[j];
                    if (j == i) {
                        // Diagonal term, only include lower-triangular
                        // component
                        hessians.push_back(casadi::SX::tril(
                            jacobian(gradient(ex, xi), xj), true));
                    } else {
                        hessians.push_back(jacobian(gradient(ex, xi), xj));
                    }
                }
            }

            // Wrap the functions
            SetHessianFunction(
                std::make_shared<utils::casadi::FunctionWrapper<MatrixType>>(
                    casadi::Function(name + "_hes", in, hessians)));
        }
    }

    std::shared_ptr<common::ScalarFunction> &ObjectiveFunction() {
        return obj_;
    }
    std::shared_ptr<common::VectorFunction> &GradientFunction() {
        return grad_;
    }
    std::shared_ptr<common::FunctionBase<MatrixType>> &HessianFunction() {
        return hes_;
    }

    /**
     * @brief Name of the cost
     *
     * @return const std::string&
     */
    const std::string &name() const { return name_; }

    /**
     * @brief Whether the cost has a Gradient
     *
     * @return true
     * @return false
     */
    const bool HasGradient() const { return has_grd_; }

    /**
     * @brief Whether the cost has a Hessian
     *
     * @return true
     * @return false
     */
    const bool HasHessian() const { return has_hes_; }

   protected:
    void SetObjectiveFunction(
        const std::shared_ptr<common::ScalarFunction> &f) {
        obj_ = f;
    }
    void SetGradientFunction(const std::shared_ptr<common::VectorFunction> &f) {
        grad_ = f;
        has_grd_ = true;
    }
    void SetHessianFunction(
        const std::shared_ptr<common::FunctionBase<MatrixType>> &f) {
        hes_ = f;
        has_hes_ = true;
    }

   private:
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
    std::shared_ptr<common::ScalarFunction> obj_;

    /**
     * @brief Gradient function
     *
     */
    std::shared_ptr<common::VectorFunction> grad_;

    /**
     * @brief Hessian function
     *
     */
    std::shared_ptr<common::FunctionBase<MatrixType>> hes_;

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

typedef CostBase<Eigen::MatrixXd> Cost;
typedef CostBase<Eigen::SparseMatrix<double>> SparseCost;

/**
 * @brief Cost of the form \f$ c^T x + b \f$
 *
 */
template <typename MatrixType>
class LinearCostBase : public CostBase<MatrixType> {
   public:
    LinearCostBase(const std::string &name, const Eigen::VectorXd &c,
               const double &b, bool jac = true)
        : CostBase<MatrixType>(name, "linear_cost") {
        // Create Costs
        casadi::DM cd, bd = b;
        damotion::utils::casadi::toCasadi(c, cd);
        casadi::SX cs = cd, bs = bd;

        ConstructConstraint(cs, bs, {}, jac, true);
    }

    LinearCostBase(const std::string &name, const casadi::SX &c,
               const casadi::SX &b, const casadi::SXVector &p, bool jac = true)
        : Cost(name, "linear_cost") {
        ConstructConstraint(c, b, p, jac, true);
    }

    LinearCostBase(const std::string &name, const sym::Expression &ex,
               bool jac = true, bool hes = true)
        : Cost(name, "linear_cost") {
        int nvar = 0;
        casadi::SXVector in = {};
        // Extract quadratic form
        casadi::SX c, b;
        casadi::SX::linear_coeff(ex, ex.Variables()[0], c, b, true);

        ConstructConstraint(c, b, ex.Parameters(), jac, hes);
    }

    /**
     * @brief Returns the coefficients of x for the cost expression.
     *
     * @return Eigen::VectorXd
     */
    const Eigen::Ref<const Eigen::VectorXd> c() { return fc_->getOutput(0); }

    /**
     * @brief Returns the constant term b in the cost expression.
     *
     * @return const double
     */
    const double b() { return fb_->getOutput(0); }

   private:
    std::shared_ptr<common::VectorFunction> fc_;
    std::shared_ptr<common::ScalarFunction> fb_;

    /**
     * @brief Compute the constraint with use of A and b
     *
     * @param input
     * @param out
     */
    void ObjectiveCallback(const common::InputRefVector &input,
                           std::vector<double> &out) {
        fc_->call(input);
        fb_->call(input);
        out[0] = fc_->getOutput(0).dot(input[0]) + fb_->getOutput(0);
    }

    /**
     * @brief Compute the gradient of the constraint with c
     *
     * @param input
     * @param out
     */
    void GradientCallback(const common::InputRefVector &input,
                          std::vector<Eigen::VectorXd> &out) {
        fc_->call(input);
        out[0] = fc_->getOutput(0);
    }

    void ConstructConstraint(const casadi::SX &c, const casadi::SX &b,
                             const casadi::SXVector &p, bool jac = true,
                             bool hes = true) {
        int nvar = 0;
        casadi::SXVector in = {};
        for (const casadi::SX &pi : p) {
            in.push_back(pi);
        }

        // Create coefficient functions
        fc_ = std::make_shared<utils::casadi::VectorFunctionWrapper>(
            casadi::Function(this->name() + "_A", in, {densify(c)}));
        fb_ = std::make_shared<utils::casadi::ScalarFunctionWrapper>(
            casadi::Function(this->name() + "_b", in, {densify(b)}));

        // Create callback functions
        std::shared_ptr<common::CallbackFunction<double>> obj_cb =
            std::make_shared<common::CallbackFunction<double>>(
                in.size(), 1,
                [this](const common::InputRefVector &in,
                       std::vector<double> &out) {
                    this->ObjectiveCallback(in, out);
                });
        std::shared_ptr<common::CallbackFunction<Eigen::VectorXd>> grd_cb =
            std::make_shared<common::CallbackFunction<Eigen::VectorXd>>(
                in.size(), 1,
                [this](const common::InputRefVector &in,
                       std::vector<Eigen::VectorXd> &out) {
                    this->GradientCallback(in, out);
                });

        // Set output sizes for the callbacks
        obj_cb->InitOutput(0, 0.0);
        grd_cb->InitOutput(0, Eigen::VectorXd::Zero(c.size1()));

        // Create functions through callbacks
        this->SetObjectiveFunction(obj_cb);
        this->SetGradientFunction(grd_cb);
    }
};

typedef LinearCostBase<Eigen::MatrixXd> LinearCost;
typedef LinearCostBase<Eigen::SparseMatrix<double>> SparseLinearCost;

/**
 * @brief A cost of the form 0.5 x^T Q x + g^T x + c
 *
 */
template <typename MatrixType>
class QuadraticCostBase : public CostBase<MatrixType> {
   public:
    QuadraticCostBase(const std::string &name, const Eigen::MatrixXd &A,
                  const Eigen::VectorXd &b, const double &c, bool jac = true,
                  bool hes = true)
        : CostBase<MatrixType>(name, "quadratic_cost") {
        // Cost
        casadi::DM Ad, bd;
        casadi::SX csx = c;
        damotion::utils::casadi::toCasadi(A, Ad);
        damotion::utils::casadi::toCasadi(b, bd);
        casadi::SX Asx = Ad, bsx = bd;
        ConstructConstraint(Asx, bsx, csx, {}, jac, hes);
    }

    QuadraticCostBase(const std::string &name, const casadi::SX &A,
                  const casadi::SX &b, const casadi::SX &c,
                  const casadi::SXVector &p, bool jac = true, bool hes = true)
        : Cost(name, "quadratic_cost") {
        ConstructConstraint(A, b, c, p, jac, hes);
    }

    QuadraticCostBase(const std::string &name, const sym::Expression &ex,
                  bool jac = true, bool hes = true)
        : Cost(name, "quadratic_cost") {
        int nvar = 0;
        casadi::SXVector in = {};
        // Extract quadratic form
        casadi::SX A, b, c;
        casadi::SX::quadratic_coeff(ex, ex.Variables()[0], A, b, c, true);

        // Remove factor of two from hessian
        A *= 0.5;

        ConstructConstraint(A, b, c, ex.Parameters(), jac, hes);
    }

    const MatrixType &A() const { return fA_->getOutput(0); }
    const Eigen::VectorXd &b() const { return fb_->getOutput(0); }
    const double &c() const { return fc_->getOutput(0); }

   private:
    std::shared_ptr<common::FunctionBase<MatrixType>> fA_;
    std::shared_ptr<common::VectorFunction> fb_;
    std::shared_ptr<common::ScalarFunction> fc_;

    /**
     * @brief Compute the cost with use of A, b and c
     *
     * @param input
     * @param out
     */
    void ObjectiveCallback(const common::InputRefVector &input,
                           std::vector<double> &out) {
        fA_->call(input);
        fb_->call(input);
        fc_->call(input);
        LOG(INFO) << "Objective";
        out[0] = input[0].dot(fA_->getOutput(0) * input[0]) +
                 fb_->getOutput(0).dot(input[0]) + fc_->getOutput(0);
        LOG(INFO) << "Objective complete";
    }

    /**
     * @brief Compute the gradient of the cost with A and b
     *
     * @param input
     * @param out
     */
    void GradientCallback(const common::InputRefVector &input,
                          std::vector<Eigen::VectorXd> &out) {
        fA_->call(input);
        fb_->call(input);
        out[0] = 2.0 * fA_->getOutput(0) * input[0] + fb_->getOutput(0);
    }

    /**
     * @brief Compute the hessian of the cost with A
     *
     * @param input
     * @param out
     */
    void HessianCallback(const common::InputRefVector &input,
                         std::vector<MatrixType> &out) {
        fA_->call(input);
        out[0] = 2.0 * fA_->getOutput(0);
    }

    void ConstructConstraint(const casadi::SX &A, const casadi::SX &b,
                             const casadi::SX &c, const casadi::SXVector &p,
                             bool jac = true, bool hes = true) {
        int nvar = 0;
        casadi::SXVector in = {};
        // Linear cost
        casadi::SX x = casadi::SX::sym("x", A.rows());
        casadi::SX cost = mtimes(x.T(), mtimes(A, x)) + mtimes(b.T(), x) + c;
        in.push_back(x);
        for (const casadi::SX &pi : p) {
            in.push_back(pi);
        }

        // Create coefficient functions
        fA_ = std::make_shared<utils::casadi::FunctionWrapper<MatrixType>>(
            casadi::Function(this->name() + "_A", in, {densify(A)}));
        fb_ = std::make_shared<utils::casadi::VectorFunctionWrapper>(
            casadi::Function(this->name() + "_b", in, {densify(b)}));
        fc_ = std::make_shared<utils::casadi::ScalarFunctionWrapper>(
            casadi::Function(this->name() + "_c", in, {densify(c)}));

        // Create callback functions
        std::shared_ptr<common::CallbackFunction<double>> obj_cb =
            std::make_shared<common::CallbackFunction<double>>(
                in.size(), 1,
                [this](const common::InputRefVector &in,
                       std::vector<double> &out) {
                    this->ObjectiveCallback(in, out);
                });
        std::shared_ptr<common::CallbackFunction<Eigen::VectorXd>> grd_cb =
            std::make_shared<common::CallbackFunction<Eigen::VectorXd>>(
                in.size(), 1,
                [this](const common::InputRefVector &in,
                       std::vector<Eigen::VectorXd> &out) {
                    this->GradientCallback(in, out);
                });
        std::shared_ptr<common::CallbackFunction<MatrixType>> hes_cb =
            std::make_shared<common::CallbackFunction<MatrixType>>(
                in.size(), 1,
                [this](const common::InputRefVector &in,
                       std::vector<MatrixType> &out) {
                    this->HessianCallback(in, out);
                });

        // Set output sizes for the callbacks
        obj_cb->InitOutput(0, 0.0);
        grd_cb->InitOutput(0, Eigen::VectorXd::Zero(b.size1()));
        hes_cb->InitOutput(0, Eigen::MatrixXd::Zero(A.size1(), A.size2()));

        // Create functions through callbacks
        this->SetObjectiveFunction(obj_cb);
        this->SetGradientFunction(grd_cb);
        this->SetHessianFunction(hes_cb);
    }
};

typedef QuadraticCostBase<Eigen::MatrixXd> QuadraticCost;
typedef QuadraticCostBase<Eigen::SparseMatrix<double>> SparseQuadraticCost;

}  // namespace optimisation
}  // namespace damotion

#endif/* SOLVERS_COST_H */
