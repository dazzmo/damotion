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
            std::make_shared<utils::casadi::FunctionWrapper<double>>(
                casadi::Function(name, in, {ex})));
        // Jacobian
        if (grd) {
            casadi::SXVector gradients;
            for (const casadi::SX &xi : ex.Variables()) {
                gradients.push_back(gradient(ex, xi));
            }
            // Wrap the functions
            SetGradientFunction(
                std::make_shared<
                    utils::casadi::FunctionWrapper<Eigen::VectorXd>>(
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

    std::shared_ptr<common::Function<double>> &ObjectiveFunction() {
        return obj_;
    }
    std::shared_ptr<common::Function<Eigen::VectorXd>> &GradientFunction() {
        return grad_;
    }
    std::shared_ptr<common::Function<MatrixType>> &HessianFunction() {
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
        const std::shared_ptr<common::Function<double>> &f) {
        obj_ = f;
    }
    void SetGradientFunction(
        const std::shared_ptr<common::Function<Eigen::VectorXd>> &f) {
        grad_ = f;
        has_grd_ = true;
    }
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
    std::shared_ptr<common::Function<double>> obj_;

    /**
     * @brief Gradient function
     *
     */
    std::shared_ptr<common::Function<Eigen::VectorXd>> grad_;

    /**
     * @brief Hessian function
     *
     */
    std::shared_ptr<common::Function<MatrixType>> hes_;

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

#endif /* SOLVERS_COST_H */
