#ifndef SOLVERS_COST_H
#define SOLVERS_COST_H

#include <casadi/casadi.hpp>

#include "symbolic/expression.h"
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

    utils::casadi::FunctionWrapper &ObjectiveFunction() { return obj_; }
    utils::casadi::FunctionWrapper &GradientFunction() { return grad_; }
    utils::casadi::FunctionWrapper &HessianFunction() { return hes_; }

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
    void SetObjectiveFunction(const casadi::Function &f) { obj_ = f; }
    void SetGradientFunction(const casadi::Function &f) {
        grad_ = f;
        has_grd_ = true;
    }
    void SetHessianFunction(const casadi::Function &f) {
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
    utils::casadi::FunctionWrapper obj_;

    /**
     * @brief Gradient function
     *
     */
    utils::casadi::FunctionWrapper grad_;

    /**
     * @brief Hessian function
     *
     */
    utils::casadi::FunctionWrapper hes_;

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
        int nvar = 0;
        casadi::SXVector in = {};
        // Constant vector b
        casadi::SX linear_cost = b;
        // Create Costs
        casadi::DM cd;
        damotion::utils::casadi::toCasadi(c, cd);
        casadi::SX cs = cd;

        casadi::SX x = casadi::SX::sym("x", c.rows());
        linear_cost += mtimes(cs.T(), x);
        in.push_back(x);

        // Create the Cost
        casadi::Function f =
            casadi::Function(this->name(), in, {linear_cost, b});
        casadi::Function fg = casadi::Function(this->name() + "_grd", in, {cs});

        SetObjectiveFunction(f);
        SetGradientFunction(fg);
    }

    LinearCost(const std::string &name, const casadi::SX &c,
               const casadi::SX &b, const casadi::SXVector &p, bool jac = true)
        : Cost(name, "linear_cost") {
        int nvar = 0;
        casadi::SXVector in = {};
        // Linear cost
        casadi::SX x = casadi::SX::sym("x", c.rows());
        casadi::SX linear_cost = mtimes(c.T(), x) + b;
        // Create Costs
        linear_cost += mtimes(c.T(), x);
        in.push_back(x);
        for (const casadi::SX &pi : p) {
            in.push_back(pi);
        }

        // Create the Cost
        casadi::Function f =
            casadi::Function(this->name(), in, {linear_cost, b});
        casadi::Function fg = casadi::Function(this->name() + "_grd", in, {c});

        SetObjectiveFunction(f);
        SetGradientFunction(fg);
    }

    LinearCost(const std::string &name, const sym::Expression &ex,
               bool jac = true, bool hes = true)
        : Cost(name, "linear_cost") {
        int nvar = 0;
        casadi::SXVector in = {};
        // Extract quadratic form
        casadi::SX c, b;
        casadi::SX::linear_coeff(ex, ex.Variables()[0], c, b, true);

        in = ex.Variables();
        for (const casadi::SX &pi : ex.Parameters()) {
            in.push_back(pi);
        }

        // Create the Cost
        casadi::Function f = casadi::Function(this->name(), in, {ex, b});
        casadi::Function fg = casadi::Function(this->name() + "_grd", in, {c});

        SetObjectiveFunction(f);
        SetGradientFunction(fg);
    }

    /**
     * @brief Returns the coefficients of x for the cost expression.
     *
     * @return Eigen::VectorXd
     */
    Eigen::VectorXd c() {
        return Eigen::Map<const Eigen::VectorXd>(
            GradientFunction().getOutput(0).data(),
            GradientFunction().getOutput(0).rows());
    }

    /**
     * @brief Returns the constant term b in the cost expression.
     *
     * @return const double
     */
    const double b() { return ObjectiveFunction().getOutput(1).data()[0]; }

   private:
};

/**
 * @brief A cost of the form 0.5 x^T Q x + g^T x + c
 *
 */
class QuadraticCost : public Cost {
   public:
    QuadraticCost(const std::string &name, const Eigen::MatrixXd &Q,
                  const Eigen::VectorXd &g, const double &c, bool jac = true,
                  bool hes = true)
        : Cost(name, "quadratic_cost") {
        int nvar = 0;
        casadi::SXVector in = {};
        // Cost
        casadi::SX cost = c;
        casadi::DM Qd, gd;
        damotion::utils::casadi::toCasadi(g, gd);
        damotion::utils::casadi::toCasadi(Q, Qd);
        casadi::SX Qs = Qd, gs = gd;
        casadi::SX x = casadi::SX::sym("x", Q.rows());
        // Create quadaratic cost expression
        cost += mtimes(x.T(), mtimes(Qs, x)) + mtimes(gs.T(), x);
        in.push_back(x);

        // Create the Cost
        casadi::Function f = casadi::Function(this->name(), in, {cost, c});
        casadi::Function fg = casadi::Function(this->name() + "_grd", in,
                                               {mtimes(Qs, x) + gs, gs});
        casadi::Function fh = casadi::Function(this->name() + "_hes", in, {Qs});

        SetObjectiveFunction(f);
        SetGradientFunction(fg);
        SetHessianFunction(fh);
    }

    QuadraticCost(const std::string &name, const casadi::SX &Q,
                  const casadi::SX &g, const casadi::SX &c,
                  const casadi::SXVector &p, bool jac = true, bool hes = true)
        : Cost(name, "quadratic_cost") {
        int nvar = 0;
        casadi::SXVector in = {};
        // Linear cost
        casadi::SX x = casadi::SX::sym("x", Q.rows());
        casadi::SX cost = mtimes(x.T(), mtimes(Q, x)) + mtimes(g.T(), x) + c;
        in.push_back(x);
        for (const casadi::SX &pi : p) {
            in.push_back(pi);
        }

        // Create the Cost
        casadi::Function f = casadi::Function(this->name(), in, {cost, c});
        casadi::Function fg =
            casadi::Function(this->name() + "_grd", in, {mtimes(Q, x) + g, g});
        casadi::Function fh = casadi::Function(this->name() + "_hes", in, {Q});

        SetObjectiveFunction(f);
        SetGradientFunction(fg);
        SetHessianFunction(fh);
    }

    QuadraticCost(const std::string &name, const sym::Expression &ex,
                  bool jac = true, bool hes = true)
        : Cost(name, "quadratic_cost") {
        int nvar = 0;
        casadi::SXVector in = {};
        // Extract quadratic form
        casadi::SX Q, g, c;
        casadi::SX::quadratic_coeff(ex, ex.Variables()[0], Q, g, c, true);

        in = ex.Variables();
        for (const casadi::SX &pi : ex.Parameters()) {
            in.push_back(pi);
        }

        // Create the Cost
        // TODO - Make only lower-triangular
        casadi::Function f = casadi::Function(this->name(), in, {ex, c});
        casadi::Function fg = casadi::Function(
            this->name() + "_grd", in, {mtimes(Q, ex.Variables()[0]) + g, g});
        casadi::Function fh = casadi::Function(this->name() + "_hes", in, {Q});

        SetObjectiveFunction(f);
        SetGradientFunction(fg);
        SetHessianFunction(fh);
    }

    const double &c() { return ObjectiveFunction().getOutput(1).data()[0]; }

    Eigen::VectorXd g() {
        return GradientFunction().getOutput(1);
    }

    const Eigen::MatrixXd &Q() { return HessianFunction().getOutput(0); }

   private:
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_COST_H */
