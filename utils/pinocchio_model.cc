#include "utils/pinocchio_model.h"

using namespace casadi_utils;

PinocchioModelWrapper &PinocchioModelWrapper::operator=(
    pinocchio::Model model) {
    // Cast model to type
    model_ = model.cast<casadi::Matrix<AD>>();
    // Create data for model
    data_ = pinocchio::DataTpl<casadi::Matrix<AD>>(model_);

    return *this;
}

casadi::Function PinocchioModelWrapper::aba() {
    // Compute expression for aba
    casadi::Matrix<AD> q = casadi::Matrix<AD>::sym("q", model_.nq),
                       v = casadi::Matrix<AD>::sym("v", model_.nv),
                       tau = casadi::Matrix<AD>::sym("tau", model_.nv), a;
    // Convert to eigen expressions
    Eigen::VectorX<casadi::Matrix<AD>> qe, ve, taue;
    eigen::toEigen(q, qe);
    eigen::toEigen(v, ve);
    eigen::toEigen(tau, taue);

    Eigen::VectorX<casadi::Matrix<AD>> ae =
        pinocchio::aba<casadi::Matrix<AD>>(model_, data_, qe, ve, taue);

    // Create AD equivalent
    eigen::toCasadi(ae, a);

    // Create function for ABA
    return casadi::Function(model_.name + "_aba", {q, v, tau}, {a},
                            {"q", "v", "u"}, {"a"});
}

casadi::Function PinocchioModelWrapper::rnea() {
    // Compute expression for aba
    casadi::Matrix<AD> q = casadi::Matrix<AD>::sym("q", model_.nq),
                       v = casadi::Matrix<AD>::sym("v", model_.nv),
                       a = casadi::Matrix<AD>::sym("a", model_.nv), u;
    // Convert to eigen expressions
    Eigen::VectorX<casadi::Matrix<AD>> qe, ve, ae;
    eigen::toEigen(q, qe);
    eigen::toEigen(v, ve);
    eigen::toEigen(a, ae);

    // Convert to SX
    Eigen::VectorX<casadi::Matrix<AD>> ue =
        pinocchio::rnea<casadi::Matrix<AD>>(model_, data_, qe, ve, ae);

    eigen::toCasadi(ue, u);

    // Create function for RNEA
    return casadi::Function(model_.name + "_rnea", {q, v, a}, {u},
                            {"q", "v", "a"}, {"u"});
};

void PinocchioModelWrapper::addEndEffector(const std::string &frame_name) {
    // Symbolic generalised coordinates and derivatives
    casadi::Matrix<AD> qpos = casadi::Matrix<AD>::sym("q", model_.nq),
                       qvel = casadi::Matrix<AD>::sym("v", model_.nv),
                       qacc = casadi::Matrix<AD>::sym("a", model_.nv);
    // Eigen-equivalents
    Eigen::VectorX<casadi::Matrix<AD>> qpos_e, qvel_e, qacc_e;
    eigen::toEigen(qpos, qpos_e);
    eigen::toEigen(qvel, qvel_e);
    eigen::toEigen(qacc, qacc_e);

    // Perform forward kinematics on model
    pinocchio::forwardKinematics(model_, data_, qpos_e, qvel_e, qacc_e);
    // Perform forward kinematics and compute frames
    pinocchio::framesForwardKinematics(model_, data_, qpos_e);

    Eigen::VectorX<casadi::Matrix<AD>> xpos_e(7), xvel_e(6), xacc_e(6);

    // Translational component
    xpos_e.topRows(3) = data_.oMf[model_.getFrameId(frame_name)].translation();
    // Rotational component
    Eigen::Matrix3<casadi::Matrix<AD>> R =
        data_.oMf[model_.getFrameId(frame_name)].rotation();
    Eigen::Quaternion<casadi::Matrix<AD>> qR;
    pinocchio::quaternion::assignQuaternion(qR, R);

    // Convert rotation matrix to quaternion representation
    xpos_e.bottomRows(4) << qR.w(), qR.vec();

    // Compute velocity of the point at the end-effector frame with respect to
    // the chosen reference frame
    xvel_e = pinocchio::getFrameVelocity(model_, data_,
                                         model_.getFrameId(frame_name),
                                         pinocchio::LOCAL_WORLD_ALIGNED)
                 .toVector();

    // Compute acceleration of the point at the end-effector frame with respect
    // to the chosen reference frame
    xacc_e = pinocchio::getFrameAcceleration(model_, data_,
                                             model_.getFrameId(frame_name),
                                             pinocchio::LOCAL_WORLD_ALIGNED)
                 .toVector();

    // Convert to casadi matrices
    casadi::Matrix<AD> xpos, xvel, xacc;
    eigen::toCasadi(xpos_e, xpos);
    eigen::toCasadi(xvel_e, xvel);
    eigen::toCasadi(xacc_e, xacc);

    // Get jacobian of this site with respect to the configuration of the
    // Compute Jacobian
    pinocchio::DataTpl<casadi::Matrix<AD>>::Matrix6x Je(6, model_.nv);
    Je.setZero();

    pinocchio::computeFrameJacobian(model_, data_, qpos_e,
                                    model_.getFrameId(frame_name),
                                    pinocchio::LOCAL_WORLD_ALIGNED, Je);

    casadi::Matrix<AD> J;
    eigen::toCasadi(Je, J);

    // Create end-effector data struct and add to vector
    EndEffector ee;
    ee.S = Eigen::Matrix<double, 6, 6>::Identity();

    ee.x = casadi::Function(model_.name + "_" + frame_name + "_ee",
                            {qpos, qvel, qacc}, {xpos, xvel, xacc},
                            {"qpos", "qvel", "qacc"}, {"xpos", "xvel", "xacc"});

    ee.J = casadi::Function(model_.name + "_" + frame_name + "_ee_jac", {qpos},
                            {J}, {"qpos"}, {"J"});

    // Add to vector
    ee_idx_[frame_name] = ee_.size();
    ee_.push_back(ee);
}
