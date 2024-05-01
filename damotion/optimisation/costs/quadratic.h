#ifndef COSTS_QUADRATIC_H
#define COSTS_QUADRATIC_H

#include "damotion/optimisation/costs/base.h"

namespace damotion {
namespace optimisation {
/**
 * @brief A cost of the form 0.5 x^T Q x + g^T x + c
 *
 */
template <typename MatrixType>
class QuadraticCost : public CostBase<MatrixType> {
   public:
    using UniquePtr = std::unique_ptr<QuadraticCost<MatrixType>>;
    using SharedPtr = std::shared_ptr<QuadraticCost<MatrixType>>;

    QuadraticCost(const std::string &name, const Eigen::MatrixXd &A,
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

    QuadraticCost(const std::string &name, const casadi::SX &A,
                  const casadi::SX &b, const casadi::SX &c,
                  const casadi::SXVector &p, bool jac = true, bool hes = true)
        : CostBase<MatrixType>(name, "quadratic_cost") {
        ConstructConstraint(A, b, c, p, jac, hes);
    }

    QuadraticCost(const std::string &name, const sym::Expression &ex,
                  bool jac = true, bool hes = true)
        : CostBase<MatrixType>(name, "quadratic_cost") {
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
    std::shared_ptr<common::Function<MatrixType>> fA_;
    std::shared_ptr<common::Function<Eigen::VectorXd>> fb_;
    std::shared_ptr<common::Function<double>> fc_;

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
        out[0] = input[0].dot(fA_->getOutput(0) * input[0]) +
                 fb_->getOutput(0).dot(input[0]) + fc_->getOutput(0);
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
        if (std::is_same<MatrixType, Eigen::SparseMatrix<double>>::value) {
            fA_ = std::make_shared<utils::casadi::FunctionWrapper<MatrixType>>(
                casadi::Function(this->name() + "_A", in, {A}));
        } else {
            fA_ = std::make_shared<utils::casadi::FunctionWrapper<MatrixType>>(
                casadi::Function(this->name() + "_A", in, {densify(A)}));
        }
        fb_ = std::make_shared<utils::casadi::FunctionWrapper<Eigen::VectorXd>>(
            casadi::Function(this->name() + "_b", in, {densify(b)}));
        fc_ = std::make_shared<utils::casadi::FunctionWrapper<double>>(
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
        obj_cb->InitialiseOutput(0, common::Sparsity());
        grd_cb->InitialiseOutput(0, common::Sparsity(b.size1(), b.size2()));
        hes_cb->InitialiseOutput(0, common::Sparsity(A.sparsity()));

        // Create functions through callbacks
        this->SetObjectiveFunction(obj_cb);
        this->SetGradientFunction(grd_cb);
        this->SetHessianFunction(hes_cb);
    }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* COSTS_QUADRATIC_H */
