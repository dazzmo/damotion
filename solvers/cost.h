#ifndef SOLVERS_COST_H
#define SOLVERS_COST_H

#include <casadi/casadi.hpp>

#include "symbolic/expression.h"
#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"

namespace damotion {
namespace optimisation {

namespace sym = damotion::symbolic;

class Cost {
   public:
    Cost() = default;
    ~Cost() = default;

    Cost(const std::string &name, const std::string &cost_type);

    Cost(const std::string &name, const symbolic::Expression &ex,
         bool grd = false, bool hes = false);

    std::shared_ptr<common::Function> &ObjectiveFunction() { return obj_; }
    std::shared_ptr<common::Function> &GradientFunction() { return grad_; }
    std::shared_ptr<common::Function> &HessianFunction() { return hes_; }

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
    void SetObjectiveFunction(const std::shared_ptr<common::Function> &f) {
        obj_ = f;
    }
    void SetGradientFunction(const std::shared_ptr<common::Function> &f) {
        grad_ = f;
        has_grd_ = true;
    }
    void SetHessianFunction(const std::shared_ptr<common::Function> &f) {
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
    std::shared_ptr<common::Function> obj_;

    /**
     * @brief Gradient function
     *
     */
    std::shared_ptr<common::Function> grad_;

    /**
     * @brief Hessian function
     *
     */
    std::shared_ptr<common::Function> hes_;

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

/**
 * @brief Cost of the form \f$ c^T x + b \f$
 *
 */
class LinearCost : public Cost {
   public:
    LinearCost(const std::string &name, const Eigen::VectorXd &c,
               const double &b, bool jac = true)
        : Cost(name, "linear_cost") {
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
    const double b() { return fb_->getOutput(0).data()[0]; }

   private:
    std::shared_ptr<common::Function> fc_;
    std::shared_ptr<common::Function> fb_;

    /**
     * @brief Compute the constraint with use of A and b
     *
     * @param input
     * @param out
     */
    void ObjectiveCallback(const common::Function::InputRefVector &input,
                           std::vector<Eigen::MatrixXd> &out) {
        fc_->call(input);
        fb_->call(input);
        out[0].data()[0] = fc_->getOutput<Eigen::VectorXd>(0).dot(input[0]) +
                           fb_->getOutput(0).data()[0];
    }

    /**
     * @brief Compute the gradient of the constraint with c
     *
     * @param input
     * @param out
     */
    void GradientCallback(const common::Function::InputRefVector &input,
                          std::vector<Eigen::MatrixXd> &out) {
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
        fc_ = std::make_shared<utils::casadi::FunctionWrapper>(
            casadi::Function(this->name() + "_A", in, {densify(c)}));
        fb_ = std::make_shared<utils::casadi::FunctionWrapper>(
            casadi::Function(this->name() + "_b", in, {densify(b)}));

        // Create callback functions
        std::shared_ptr<common::CallbackFunction> obj_cb =
            std::make_shared<common::CallbackFunction>(
                in.size(), 1,
                [this](const common::Function::InputRefVector &in,
                       std::vector<Eigen::MatrixXd> &out) {
                    this->ObjectiveCallback(in, out);
                });
        std::shared_ptr<common::CallbackFunction> grd_cb =
            std::make_shared<common::CallbackFunction>(
                in.size(), 1,
                [this](const common::Function::InputRefVector &in,
                       std::vector<Eigen::MatrixXd> &out) {
                    this->GradientCallback(in, out);
                });

        // Set output sizes for the callbacks
        obj_cb->setOutputSize(0, 1, 1);
        grd_cb->setOutputSize(0, c.size1(), 1);

        // Create functions through callbacks
        SetObjectiveFunction(obj_cb);
        SetGradientFunction(grd_cb);
    }
};

/**
 * @brief A cost of the form 0.5 x^T Q x + g^T x + c
 *
 */
class QuadraticCost : public Cost {
   public:
    QuadraticCost(const std::string &name, const Eigen::MatrixXd &A,
                  const Eigen::VectorXd &b, const double &c, bool jac = true,
                  bool hes = true)
        : Cost(name, "quadratic_cost") {
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
        : Cost(name, "quadratic_cost") {
        ConstructConstraint(A, b, c, p, jac, hes);
    }

    QuadraticCost(const std::string &name, const sym::Expression &ex,
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

    const Eigen::Ref<const Eigen::MatrixXd> A() { return fA_->getOutput(0); }
    const Eigen::Ref<const Eigen::VectorXd> b() {
        return fb_->getOutput<Eigen::VectorXd>(0);
    }
    const double &c() { return fc_->getOutput(0).data()[0]; }

   private:
    std::shared_ptr<common::Function> fA_;
    std::shared_ptr<common::Function> fb_;
    std::shared_ptr<common::Function> fc_;

    /**
     * @brief Compute the cost with use of A, b and c
     *
     * @param input
     * @param out
     */
    void ObjectiveCallback(const common::Function::InputRefVector &input,
                           std::vector<Eigen::MatrixXd> &out) {
        fA_->call(input);
        fb_->call(input);
        fc_->call(input);
        LOG(INFO) << "Objective";
        out[0].data()[0] = input[0].dot(fA_->getOutput(0) * input[0]) +
                           fb_->getOutput<Eigen::VectorXd>(0).dot(input[0]) +
                           fc_->getOutput(0).data()[0];
        LOG(INFO) << "Objective complete";
    }

    /**
     * @brief Compute the gradient of the cost with A and b
     *
     * @param input
     * @param out
     */
    void GradientCallback(const common::Function::InputRefVector &input,
                          std::vector<Eigen::MatrixXd> &out) {
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
    void HessianCallback(const common::Function::InputRefVector &input,
                         std::vector<Eigen::MatrixXd> &out) {
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
        fA_ = std::make_shared<utils::casadi::FunctionWrapper>(
            casadi::Function(this->name() + "_A", in, {densify(A)}));
        fb_ = std::make_shared<utils::casadi::FunctionWrapper>(
            casadi::Function(this->name() + "_b", in, {densify(b)}));
        fc_ = std::make_shared<utils::casadi::FunctionWrapper>(
            casadi::Function(this->name() + "_c", in, {densify(c)}));

        // Create callback functions
        std::shared_ptr<common::CallbackFunction> obj_cb =
            std::make_shared<common::CallbackFunction>(
                in.size(), 1,
                [this](const common::Function::InputRefVector &in,
                       std::vector<Eigen::MatrixXd> &out) {
                    this->ObjectiveCallback(in, out);
                });
        std::shared_ptr<common::CallbackFunction> grd_cb =
            std::make_shared<common::CallbackFunction>(
                in.size(), 1,
                [this](const common::Function::InputRefVector &in,
                       std::vector<Eigen::MatrixXd> &out) {
                    this->GradientCallback(in, out);
                });
        std::shared_ptr<common::CallbackFunction> hes_cb =
            std::make_shared<common::CallbackFunction>(
                in.size(), 1,
                [this](const common::Function::InputRefVector &in,
                       std::vector<Eigen::MatrixXd> &out) {
                    this->HessianCallback(in, out);
                });

        // Set output sizes for the callbacks
        obj_cb->setOutputSize(0, 1, 1);
        grd_cb->setOutputSize(0, b.size1(), 1);
        hes_cb->setOutputSize(0, A.size1(), A.size2());

        // Create functions through callbacks
        SetObjectiveFunction(obj_cb);
        SetGradientFunction(grd_cb);
        SetHessianFunction(hes_cb);
    }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_COST_H */
