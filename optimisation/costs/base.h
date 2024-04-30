#ifndef COSTS_BASE_H
#define COSTS_BASE_H

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

        // Convert input variables to single vector
        casadi::SX x = casadi::SX::vertcat(ex.Variables());

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
            std::make_shared<utils::casadi::FunctionWrapper<double>>(
                casadi::Function(name, in, {ex})));
        // Jacobian
        if (grd) {
            // Wrap the functions
            SetGradientFunction(
                std::make_shared<
                    utils::casadi::FunctionWrapper<Eigen::VectorXd>>(
                    casadi::Function(name + "_grd", in, gradient(ex, x))));
        }

        // Hessian
        if (hes) {
            // Wrap the functions
            SetHessianFunction(
                std::make_shared<utils::casadi::FunctionWrapper<MatrixType>>(
                    casadi::Function(name + "_hes", in,
                                     casadi::SX::tril(hessian(ex, x)))));
        }
    }

    const std::shared_ptr<common::Function<double>> &ObjectiveFunction() const {
        return obj_;
    }
    const std::shared_ptr<common::Function<Eigen::VectorXd>> &GradientFunction()
        const {
        return grad_;
    }
    const std::shared_ptr<common::Function<MatrixType>> &HessianFunction()
        const {
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

    /**
     * @brief Evaluate the cost with the current input variables and
     * parameters, indicating if gradient is required
     *
     * @param x Variables for the cost
     * @param p Parameters for the cost
     * @param grd Whether to also compute the gradient
     */
    virtual void eval(const Eigen::VectorXd &x, const common::InputRefVector &p,
                      bool grd = true) const {
        VLOG(10) << this->name() << " eval()";
        common::InputRefVector in = {x};
        for (int i = 0; i < p.size(); ++i) in.push_back(p[i]);

        // Call necessary cost functions
        this->obj_->call(in);
        if (grd) this->grd_->call(in);
    }

    /**
     * @brief Evaluate the Hessian of the constraint with respec to the inputs x
     *
     * @param x
     * @param p
     */
    void eval_hessian(const Eigen::VectorXd &x, const Eigen::VectorXd &l,
                      const common::InputRefVector &p) {
        // Create input for the lambda-hessian product
        common::InputRefVector in = {x, l};
        for (int i = 0; i < p.size(); ++i) in.push_back(p[i]);
        // Call necessary constraint functions
        this->hes_->call(in);
    }

    /**
     * @brief Returns the most recent evaluation of the cost objective
     *
     * @return const double&
     */
    virtual const double &Objective() const { return obj_->getOutput(0); }
    /**
     * @brief The gradient of the cost with respect to the i-th variable
     * vector
     *
     * @param i
     * @return const VectorXd&
     */
    virtual const Eigen::VectorXd &Gradient() const {
        return grd_->getOutput(0);
    }
    /**
     * @brief Returns the Hessian block with respect to the variables.
     * Please note that this formulation produces only the lower-triangular.
     *
     * @param i
     * @param j
     * @return const MatrixType&
     */
    const MatrixType &Hessian() const { return hes_->getOutput(0); }

    /**
     * @brief Number of parameters used to determine the constraint
     *
     * @return const int&
     */
    const int &NumberOfInputParameters() const { return np_; }

   protected:
    /**
     * @brief Set the Objective Function object
     *
     * @param f
     */
    void SetObjectiveFunction(
        const std::shared_ptr<common::Function<double>> &f) {
        obj_ = f;
    }

    /**
     * @brief Set the Gradient Function object
     *
     * @param f
     */
    void SetGradientFunction(
        const std::shared_ptr<common::Function<Eigen::VectorXd>> &f) {
        grad_ = f;
        has_grd_ = true;
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
    common::Function<double>::SharedPtr obj_;

    /**
     * @brief Gradient function
     *
     */
    common::Function<Eigen::VectorXd>::SharedPtr grad_;

    /**
     * @brief Hessian function
     *
     */
    common::Function<MatrixType>::SharedPtr hes_;

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

}  // namespace optimisation
}  // namespace damotion

#endif/* COSTS_BASE_H */
