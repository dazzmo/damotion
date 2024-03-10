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

    // Include external forces if end-effectors are included
    int ne = ee_.size();
    if (ne) {
        // Determine for each end-effector how many contact forces to include
        int nc = 0;
        for (int i = 0; i < ee_.size(); ++i) {
            nc += ee_constraint_subspace_[i].count();
        }
        // Create constraint force vector
        casadi::Matrix<AD> f = casadi::Matrix<AD>::sym("f", nc);
        int idx = 0;
        // Augment generalised input tau to also include constraint forces in
        // joint-space
        for (int i = 0; i < ee_.size(); ++i) {
            casadi::Function &J = ee_jac_[i];
            casadi::Matrix<AD> Jc = J({q})[0];
            // Get slice of constraint forces for end effector
            int nci = ee_constraint_subspace_[i].count();
            casadi::Matrix<AD> fi = f(casadi::Slice(idx, idx + nci));
            // Determine joint-space forces and add to generalised input
            tau += mtimes(Jc.T(), fi);
            // Increase index in force vector
            idx += nci;
        }
        // Create function for ABA
        return casadi::Function(model_.name + "_aba", {q, v, tau, f}, {a},
                                {"q", "v", "u", "f"}, {"a"});
    } else {
        // Create function for ABA
        return casadi::Function(model_.name + "_aba", {q, v, tau}, {a},
                                {"q", "v", "u"}, {"a"});
    }
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

    // Include external forces if end-effectors are included
    int ne = ee_.size();
    if (ne) {
        // Determine for each end-effector how many contact forces to include
        int nc = 0;
        for (int i = 0; i < ee_.size(); ++i) {
            nc += ee_constraint_subspace_[i].cols();
        }
        // Create constraint force vector
        casadi::Matrix<AD> f = casadi::Matrix<AD>::sym("f", nc);

        int idx = 0;
        for (int i = 0; i < ee_.size(); ++i) {
            casadi::Function &J = ee_jac_[i];
            casadi::Matrix<AD> Jc = J({q})[0];
            // Get slice of constraint forces for end effector
            int nci = ee_constraint_subspace_[i].cols();
            casadi::DM Sd;
            eigen::toCasadi(ee_constraint_subspace_[i], Sd);
            casadi::Matrix<AD> Sc = Sd;
            casadi::Matrix<AD> fi = f(casadi::Slice(idx, idx + nci));
            // Determine joint-space forces
            u -= mtimes(mtimes(Jc.T(), Sc), fi);
            // Increase index in force vector
            idx += nci;
        }
        // Create function for RNEA
        return casadi::Function(model_.name + "_rnea", {q, v, a, f}, {u},
                                {"q", "v", "a", "f"}, {"u"});
    } else {
        // Create function for RNEA
        return casadi::Function(model_.name + "_rnea", {q, v, a}, {u},
                                {"q", "v", "a"}, {"u"});
    }
};

// ! Create selector MATRIX S which will be at most 6 x 6
void PinocchioModelWrapper::setEndEffectorConstraintSubspace(
    int i, const Eigen::Matrix<double, 6, -1> &S) {
    ee_constraint_subspace_[i] = S;
};

void PinocchioModelWrapper::addEndEffector(const std::string &frame_name) {
    // Create function with custom jacobian and time derivative
    casadi::Matrix<AD> q = casadi::Matrix<AD>::sym("q", model_.nq),
                       v = casadi::Matrix<AD>::sym("v", model_.nv),
                       x = casadi::Matrix<AD>::sym("x", 6);
    Eigen::VectorX<casadi::Matrix<AD>> qe, ve;
    eigen::toEigen(q, qe);
    eigen::toEigen(v, ve);

    // Perform forward kinematics and compute frames
    pinocchio::framesForwardKinematics(model_, data_, qe);
    // Get position of the end-effector in the desired frame
    Eigen::VectorX<casadi::Matrix<AD>> xe(7);
    // Convert position and orientation of point to translation and quaternion
    xe.topRows(3) = data_.oMf[model_.getFrameId(frame_name)].translation();
    Eigen::Matrix3<casadi::Matrix<AD>> R =
        data_.oMf[model_.getFrameId(frame_name)].rotation();

    Eigen::Quaternion<casadi::Matrix<AD>> qR;
    pinocchio::quaternion::assignQuaternion(qR, R);
    // Convert rotation matrix to quaternion representation
    xe.bottomRows(4) << qR.w(), qR.vec();

    eigen::toCasadi(xe, x);

    // Get jacobian of this site with respect to the configuration of the
    // Compute Jacobian
    pinocchio::DataTpl<casadi::Matrix<AD>>::Matrix6x Je(6, model_.nv);
    Je.setZero();
    // pinocchio::computeFrameJacobian()
    pinocchio::computeFrameJacobian(model_, data_, qe,
                                    model_.getFrameId(frame_name),
                                    pinocchio::LOCAL_WORLD_ALIGNED, Je);

    casadi::Matrix<AD> J;
    eigen::toCasadi(Je, J);

    // Add full constraint subspace unless otherwise stated
    ee_constraint_subspace_.push_back(Eigen::Vector<double, 6>::Ones());

    // Create functions for end effector details
    ee_.push_back(
        casadi::Function(model_.name + "_" + frame_name + "_end_effector", {q},
                         {x}, {"q"}, {"x"}));

    ee_jac_.push_back(
        casadi::Function(model_.name + "_" + frame_name + "_end_effector_jac",
                         {q}, {J}, {"q"}, {"J"}));
}
