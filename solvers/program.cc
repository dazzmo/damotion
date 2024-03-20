#include "solvers/program.h"

using namespace damotion::solvers;

Program::Cost::Cost(const std::string &name, casadi::Function &c,
                    const casadi::SX &x) {
    // Create name vector
    casadi::SXVector in, out;
    inames = {};
    for (int i = 0; i < c.n_in(); ++i) {
        inames.push_back(c.name_in(i));
        in.push_back(casadi::SX::sym(c.name_in(i), c.size1_in(i)));
    }

    // Evaluate objective with given inputs
    casadi::SX obj = c(in)[0];

    // Create functions to compute the necessary gradients and hessians
    
    /* Objective */
    casadi::Function co =
        casadi::Function(name + "_obj", in, {obj}, inames, {name + "_obj"});
    this->obj = eigen::FunctionWrapper(c);

    /* Objective Gradient */
    casadi::Function cg = casadi::Function(
        name + "_grad", in, {gradient(obj, x)}, inames, {name + "_grad"});
    this->grad = eigen::FunctionWrapper(cg);

    /* Objective Hessian */
    casadi::Function ch = casadi::Function(name + "_hes", in, {hessian(obj, x)},
                                           inames, {name + "_hes"});
    this->hes = eigen::FunctionWrapper(ch);
}

Program::Constraint::Constraint(const std::string &name, casadi::Function &c,
                                const casadi::SX &x) : name_(name) {
    // Create name vector
    casadi::SXVector in, out;
    inames = {};
    for (int i = 0; i < c.n_in(); ++i) {
        inames.push_back(c.name_in(i));
        in.push_back(casadi::SX::sym(c.name_in(i), c.size1_in(i)));
    }

    // Evaluate constraints
    casadi::SX constraint = c(in)[0];

    // Create functions to compute the necessary gradients and hessians
    casadi::Function co = casadi::Function(name + "_con", in, {constraint},
                                           inames, {name + "_con"});
    this->con = eigen::FunctionWrapper(co);

    /* Constraint Gradient */
    casadi::Function cjac = casadi::Function(
        name + "_jac", in, {jacobian(constraint, x)}, inames, {name + "_jac"});
    this->jac = eigen::FunctionWrapper(cjac);

    // Set dimension of constraint
    dim_ = constraint.size1();

    // Create generic bounds
    double inf = std::numeric_limits<double>::infinity();
    ub_ = inf * Eigen::VectorXd::Ones(dim_);
    lb_ = -inf * Eigen::VectorXd::Ones(dim_);
}