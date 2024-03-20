#include "solvers/program.h"

namespace damotion {

namespace solvers {

Program::Cost::Cost(const std::string &name, casadi::SX &f,
                    const casadi::SXVector &in,
                    const casadi::StringVector &inames, const casadi::SX &x) {
    // Create functions to compute the necessary gradients and hessians
    /* Objective */
    casadi::Function obj =
        casadi::Function(name + "_obj", in, {f}, inames, {name + "_obj"});
    this->obj = eigen::FunctionWrapper(obj);

    /* Objective Gradient */
    casadi::Function grad = casadi::Function(
        name + "_grad", in, {gradient(f, x)}, inames, {name + "_grad"});
    this->grad = eigen::FunctionWrapper(grad);

    /* Objective Hessian */
    casadi::Function hes = casadi::Function(name + "_hes", in, {hessian(f, x)},
                                            inames, {name + "_hes"});
    this->hes = eigen::FunctionWrapper(hes);
}

Program::Constraint::Constraint(const std::string &name, casadi::SX &c,
                                const casadi::SXVector &in,
                                const casadi::StringVector &inames,
                                const casadi::SX &x)
    : name_(name) {
    // Create functions to compute the necessary gradients and hessians

    /* Objective */
    casadi::Function con =
        casadi::Function(name + "_con", in, {c}, inames, {name + "_con"});
    this->con = eigen::FunctionWrapper(con);

    /* Objective Gradient */
    casadi::Function jac = casadi::Function(name + "_jac", in, {jacobian(c, x)},
                                            inames, {name + "_jac"});
    this->jac = eigen::FunctionWrapper(jac);

    // Set dimension of constraint
    dim_ = c.size1();

    // Create generic bounds
    double inf = std::numeric_limits<double>::infinity();
    ub_ = inf * Eigen::VectorXd::Ones(dim_);
    lb_ = -inf * Eigen::VectorXd::Ones(dim_);
}

}  // namespace solvers
}  // namespace damotion