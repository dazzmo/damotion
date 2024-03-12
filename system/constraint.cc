#include "system/constraint.h"

damotion::system::HolonomicConstraint::HolonomicConstraint(
    const std::string &name,
    casadi_utils::PinocchioModelWrapper::EndEffector &ee)
    : Constraint(name, ee.S.cols()),
      nq_(ee.x.size1_in(ee.x.index_in("qpos"))),
      nv_(ee.x.size1_in(ee.x.index_in("qvel"))) {
    // Create inputs
    casadi::SX q = casadi::SX::sym("qpos", ee.x.size1_in(0));
    casadi::SX v = casadi::SX::sym("qvel", ee.x.size1_in(1));
    casadi::SX a = casadi::SX::sym("qacc", ee.x.size1_in(2));
    casadi::SXVector in, out;

    // Evaluate end-effector kinematics
    in = {q, v, a};
    out = ee.x(in);
    
    // Compute constrained outputs
    casadi::DM Sd;
    casadi_utils::eigen::toCasadi(ee.S, Sd);
    casadi::SX S = Sd;
    casadi::SX xc = mtimes(S.T(), out[0]);
    casadi::SX xvc = mtimes(S.T(), out[1]);
    casadi::SX xac = mtimes(S.T(), out[2]);

    // Also compute jacobian in constrained directions
    in = {q};
    out = ee.J(in);
    casadi::SX Jc = mtimes(S.T(), Jc);

    // Compute constraint, Jacobian and second time derivative
    setConstraint(casadi::Function(name + "_c", {q}, {xc}, {"qpos"}, {"c"}));
    setJacobian(casadi::Function(name + "_jac", {q}, {Jc}, {"qpos"}, {"J"}));
    setFirstTimeDerivative(casadi::Function(name + "_dcdt", {q, v}, {xvc},
                                            {"qpos", "qvel"}, {"dcdt"}));
    setSecondTimeDerivative(casadi::Function(name + "_d2cdt2", {q, v, a}, {xac},
                                             {"qpos", "qvel", "qacc"},
                                             {"d2cdt2"}));
}

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