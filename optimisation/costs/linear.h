#ifndef COSTS_LINEAR_H
#define COSTS_LINEAR_H

#include "optimisation/costs/base.h"

namespace damotion {
namespace optimisation {
/**
 * @brief Cost of the form \f$ c^T x + b \f$
 *
 */
template <typename MatrixType>
class LinearCost : public CostBase<MatrixType> {
   public:
    LinearCost(const std::string &name, const Eigen::VectorXd &c,
               const double &b, bool jac = true)
        : CostBase<MatrixType>(name, "linear_cost") {
        // Create Costs
        casadi::DM cd, bd = b;
        damotion::utils::casadi::toCasadi(c, cd);
        casadi::SX cs = cd, bs = bd;

        ConstructConstraint(cs, bs, {}, jac, true);
    }

    LinearCost(const std::string &name, const casadi::SX &c,
               const casadi::SX &b, const casadi::SXVector &p, bool jac = true)
        : Cost(name, "linear_cost") {
        ConstructConstraint(c, b, p, jac, true);
    }

    LinearCost(const std::string &name, const sym::Expression &ex,
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
    std::shared_ptr<common::Function<Eigen::VectorXd>> fc_;
    std::shared_ptr<common::Function<double>> fb_;

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
        fc_ = std::make_shared<utils::casadi::FunctionWrapper<Eigen::VectorXd>>(
            casadi::Function(this->name() + "_A", in, {densify(c)}));
        fb_ = std::make_shared<utils::casadi::FunctionWrapper<double>>(
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
        obj_cb->InitialiseOutput(0, common::Sparsity());
        grd_cb->InitialiseOutput(0, common::Sparsity(c.size1(), c.size2()));

        // Create functions through callbacks
        this->SetObjectiveFunction(obj_cb);
        this->SetGradientFunction(grd_cb);
    }
};

}  // namespace optimisation

}  // namespace damotion

#endif /* COSTS_LINEAR_H */
