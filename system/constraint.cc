#include "system/constraint.h"

using namespace damotion::system;

casadi::Function constrainedDynamics(
    SecondOrderControlledSystem &system,
    std::vector<HolonomicConstraint> &constraints) {
    // For each constraint, compute tangent-space forces by Jacobian
    casadi::SX f = casadi::SX::zeros(system.nv());

    // Create symbolic vectors
    casadi::SX q = casadi::SX::sym("q", system.nq());
    casadi::SX v = casadi::SX::sym("v", system.nv());
    casadi::SX tau = casadi::SX::sym("tau", system.nv());
    // Constraint forces
    casadi::SX f = casadi::SX::sym("f", system.nv());

    int idx = 0;
    for (auto &c : constraints) {
        casadi::SX Jc = c.jac()({q})[0];
        int nci = c.nc();
        casadi::SX fi = f(casadi::Slice(idx, idx + nci));
        // Determine joint-space forces and add to generalised input
        tau += mtimes(Jc.T(), fi);
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

casadi::Function constrainedInverseDynamics(
    SecondOrderControlledSystem &system,
    std::vector<HolonomicConstraint> &constraints) {
    // For each constraint, compute tangent-space forces by Jacobian
    casadi::SX f = casadi::SX::zeros(system.nv());

    // Create symbolic vectors
    casadi::SX q = casadi::SX::sym("q", system.nq());
    casadi::SX v = casadi::SX::sym("v", system.nv());
    casadi::SX a = casadi::SX::sym("a", system.nv());
    // Constraint forces
    casadi::SX f = casadi::SX::sym("f", system.nv());

    // Compute constraint-free input
    // Compute forward dynamics as a result
    casadi::SX tau = casadi::SX::zeros(system.nv());
    casadi::SXVector in = {q, v, a};
    tau = system.inverseDynamics()(in)[0];

    int idx = 0;
    for (auto &c : constraints) {
        casadi::SX Jc = c.jac()({q})[0];
        int nci = c.nc();
        casadi::SX fi = f(casadi::Slice(idx, idx + nci));
        // Determine joint-space forces and add to generalised input
        tau -= mtimes(Jc.T(), fi);
    }

    // Create new constrained dynamics with constraint forces as input
    return casadi::Function(system.inverseDynamics().name() + "_constrained",
                            {q, v, a, f}, {tau}, {"q", "v", "a", "f"}, {"tau"});
}