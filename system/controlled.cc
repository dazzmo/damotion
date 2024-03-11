#include "system/controlled.h"

using namespace damotion::system;

SecondOrderControlledSystem &SecondOrderControlledSystem::operator=(
    casadi_utils::PinocchioModelWrapper model) {
    // Create a SecondOrderControlledSystem class from the model wrapper
    this->nq_ = model.model().nq;
    this->nv_ = model.model().nv;

    // Create functions to evaluate the dynamics
    casadi::Function aba = model.aba();
    casadi::SX q = casadi::SX::sym("q", model.model().nq);
    casadi::SX v = casadi::SX::sym("v", model.model().nv);
    casadi::SX tau = casadi::SX::sym("tau", model.model().nv);
    casadi::SX a;
    // aba.call({q, v, tau}, {a});

    // Create forward dynamics function
    casadi::SX x = casadi::SX::vertcat({q, v});
    casadi::SX dx = casadi::SX::vertcat({v, a});
    setDynamics(casadi::Function(model.model().name + "_fwd", {x, tau}, {dx}));

    // Create inverse dynamics function
    setInverseDynamics(model.rnea());

    return *this;
}