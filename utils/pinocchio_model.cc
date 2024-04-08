#include "utils/pinocchio_model.h"

namespace damotion {
namespace utils {
namespace casadi {

PinocchioModelWrapper &PinocchioModelWrapper::operator=(
    pinocchio::Model model) {
    // Cast model to type
    model_ = model.cast<::casadi::Matrix<AD>>();
    // Create data for model
    data_ = pinocchio::DataTpl<::casadi::Matrix<AD>>(model_);

    return *this;
}

::casadi::Function PinocchioModelWrapper::aba() {
    // Compute expression for aba
    ::casadi::Matrix<AD> q = ::casadi::Matrix<AD>::sym("q", model_.nq),
                         v = ::casadi::Matrix<AD>::sym("v", model_.nv),
                         tau = ::casadi::Matrix<AD>::sym("tau", model_.nv), a;
    // Convert to eigen expressions
    Eigen::VectorX<::casadi::Matrix<AD>> qe, ve, taue;
    toEigen(q, qe);
    toEigen(v, ve);
    toEigen(tau, taue);

    Eigen::VectorX<::casadi::Matrix<AD>> ae =
        pinocchio::aba<::casadi::Matrix<AD>>(model_, data_, qe, ve, taue);

    // Create AD equivalent
    toCasadi(ae, a);

    // Create function for ABA
    return ::casadi::Function(model_.name + "_aba", {q, v, tau}, {a},
                              {"q", "v", "u"}, {"a"});
}

::casadi::Function PinocchioModelWrapper::rnea() {
    // Compute expression for aba
    ::casadi::Matrix<AD> q = ::casadi::Matrix<AD>::sym("q", model_.nq),
                         v = ::casadi::Matrix<AD>::sym("v", model_.nv),
                         a = ::casadi::Matrix<AD>::sym("a", model_.nv), u;
    // Convert to eigen expressions
    Eigen::VectorX<::casadi::Matrix<AD>> qe, ve, ae;
    toEigen(q, qe);
    toEigen(v, ve);
    toEigen(a, ae);

    // Convert to SX
    Eigen::VectorX<::casadi::Matrix<AD>> ue =
        pinocchio::rnea<::casadi::Matrix<AD>>(model_, data_, qe, ve, ae);

    toCasadi(ue, u);

    // Create function for RNEA
    return ::casadi::Function(model_.name + "_rnea", {q, v, a}, {u},
                              {"q", "v", "a"}, {"u"});
};

void PinocchioModelWrapper::addEndEffector(const std::string &frame_name) {
    // Symbolic generalised coordinates and derivatives
    ::casadi::Matrix<AD> qpos = ::casadi::Matrix<AD>::sym("q", model_.nq),
                         qvel = ::casadi::Matrix<AD>::sym("v", model_.nv),
                         qacc = ::casadi::Matrix<AD>::sym("a", model_.nv);
    // Eigen-equivalents
    Eigen::VectorX<::casadi::Matrix<AD>> qpos_e, qvel_e, qacc_e;
    toEigen(qpos, qpos_e);
    toEigen(qvel, qvel_e);
    toEigen(qacc, qacc_e);

    // Perform forward kinematics on model
    pinocchio::forwardKinematics(model_, data_, qpos_e, qvel_e, qacc_e);
    // Get SE3 data for the target frame
    pinocchio::SE3Tpl<::casadi::Matrix<AD>> se3_frame =
        data_.oMf[model_.getFrameId(frame_name)];

    Eigen::VectorX<::casadi::Matrix<AD>> pos_e(7), vel_e(6), acc_e(6);

    // Translational component
    pos_e.topRows(3) = se3_frame.translation();
    // Rotational component
    Eigen::Matrix3<::casadi::Matrix<AD>> R = se3_frame.rotation();
    Eigen::Quaternion<::casadi::Matrix<AD>> qR;
    pinocchio::quaternion::assignQuaternion(qR, R);

    // Convert rotation matrix to quaternion representation
    pos_e.bottomRows(4) << qR.w(), qR.vec();

    // Compute velocity of the point at the end-effector frame with respect to
    // the chosen reference frame
    vel_e = pinocchio::getFrameVelocity(model_, data_,
                                        model_.getFrameId(frame_name),
                                        pinocchio::LOCAL_WORLD_ALIGNED)
                .toVector();

    // Compute acceleration of the point at the end-effector frame with respect
    // to the chosen reference frame
    acc_e = pinocchio::getFrameClassicalAcceleration(
                model_, data_, model_.getFrameId(frame_name),
                pinocchio::LOCAL_WORLD_ALIGNED)
                .toVector();

    // Convert to casadi matrices
    ::casadi::Matrix<AD> pos, vel, acc;
    toCasadi(pos_e, pos);
    toCasadi(vel_e, vel);
    toCasadi(acc_e, acc);

    // Get jacobian of this site with respect to the configuration of the
    // Compute Jacobian
    pinocchio::DataTpl<::casadi::Matrix<AD>>::Matrix6x Je(6, model_.nv);
    Je.setZero();

    pinocchio::computeFrameJacobian(model_, data_, qpos_e,
                                    model_.getFrameId(frame_name),
                                    pinocchio::LOCAL_WORLD_ALIGNED, Je);

    ::casadi::Matrix<AD> J;
    toCasadi(Je, J);

    // Create end-effector data struct and add to vector
    EndEffector ee;

    ee.x = ::casadi::Function(model_.name + "_" + frame_name + "_ee",
                              {qpos, qvel, qacc},
                              {densify(pos), densify(vel), densify(acc)},
                              {"qpos", "qvel", "qacc"}, {"pos", "vel", "acc"});

    ee.J = ::casadi::Function(model_.name + "_" + frame_name + "_ee_jac",
                              {qpos}, {J}, {"qpos"}, {"J"});

    // Add to map
    ee_idx_[frame_name] = ee_.size();
    ee_.push_back(ee);
}

}  // namespace casadi
}  // namespace utils
}  // namespace damotion
