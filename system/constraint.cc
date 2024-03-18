#include "system/constraint.h"

damotion::system::Constraint::Constraint(const std::string &name,
                                         const casadi::Function &constraint,
                                         const casadi::Function &jacobian)
    : name_(name), f_(constraint), df_(jacobian) {
    nc_ = constraint.size1_out(0);
}

void damotion::system::Constraint::setConstraint(
    const casadi::SX &c, const casadi::SXVector &in,
    const casadi::StringVector &name_in, const casadi::StringVector &name_out) {
    // Create function based on inputs
    setConstraint(casadi::Function(name(), in, {c}, name_in, name_out));
}

void damotion::system::Constraint::setJacobian(const casadi::SX &J,
                                               const casadi::SXVector &in) {}

damotion::system::HolonomicConstraint::HolonomicConstraint(
    const std::string &name, const casadi::Function &constraint,
    const casadi::Function &jacobian,
    const casadi::Function &first_time_derivative,
    const casadi::Function &second_time_derivative)
    : Constraint(name, constraint, jacobian),
      df_(first_time_derivative),
      ddf_(second_time_derivative) {
    nq_ = constraint.size1_in("qpos");
    nv_ = first_time_derivative.size1_in("qvel");
}

damotion::system::HolonomicConstraint::HolonomicConstraint(
    const std::string &name, const casadi::SX &c, const casadi::SX &dc,
    const casadi::SX &ddc, const casadi::SX &J, const casadi::SX &qpos,
    const casadi::SX &qvel, const casadi::SX &qacc,
    const casadi::SXVector &par) {}

casadi::Function damotion::system::constrainedDynamics(
    SecondOrderControlledSystem &system,
    std::vector<HolonomicConstraint> &constraints) {
    // Create symbolic vectors
    casadi::SX q = casadi::SX::sym("q", system.nq());
    casadi::SX v = casadi::SX::sym("v", system.nv());
    casadi::SX tau = casadi::SX::sym("tau", system.nv());
    // Constraint forces
    int nc = 0;
    for (auto &c : constraints) {
        nc += c.nc();
    }
    casadi::SX f = casadi::SX::sym("f", nc);

    int idx = 0;
    for (auto &c : constraints) {
        casadi::SX Jc = c.jacobian()({q})[0];
        int nci = c.nc();
        casadi::SX fi = f(casadi::Slice(idx, idx + nci));
        // Determine joint-space forces and add to generalised input
        tau += mtimes(Jc.T(), fi);
        idx += nci;
    }

    // Create input state
    casadi::SX x = casadi::SX::vertcat({q, v});
    // Compute forward dynamics as a result
    casadi::SX a = casadi::SX::zeros(system.nv());
    casadi::SXVector in = {x, tau};
    a = system.dynamics()(in)[0];

    // Create output state
    casadi::SX dx = casadi::SX::vertcat({v, a});

    // Create new constrained dynamics with constraint forces as input
    return casadi::Function(system.dynamics().name() + "_constrained",
                            {x, tau, f}, {dx}, {"x", "tau", "f"}, {"dx"});
}

casadi::Function damotion::system::constrainedInverseDynamics(
    SecondOrderControlledSystem &system,
    std::vector<HolonomicConstraint> &constraints) {
    // Create symbolic vectors
    casadi::SX q = casadi::SX::sym("q", system.nq());
    casadi::SX v = casadi::SX::sym("v", system.nv());
    casadi::SX a = casadi::SX::sym("a", system.nv());
    // Constraint forces
    int nc = 0;
    for (auto &c : constraints) {
        nc += c.nc();
    }
    casadi::SX f = casadi::SX::sym("f", nc);

    // Compute constraint-free input
    // Compute forward dynamics as a result
    casadi::SX tau = casadi::SX::zeros(system.nv());
    casadi::SXVector in = {q, v, a};
    tau = system.inverseDynamics()(in)[0];

    int idx = 0;
    for (auto &c : constraints) {
        casadi::SX Jc = c.jacobian()({q})[0];
        int nci = c.nc();
        casadi::SX fi = f(casadi::Slice(idx, idx + nci));
        // Determine joint-space forces and add to generalised input
        tau -= mtimes(Jc.T(), fi);
        idx += nci;
    }

    // Create new constrained dynamics with constraint forces as input
    return casadi::Function(system.inverseDynamics().name() + "_constrained",
                            {q, v, a, f}, {tau}, {"q", "v", "a", "f"}, {"tau"});
}