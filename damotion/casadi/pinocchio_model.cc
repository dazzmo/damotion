#include "damotion/casadi/pinocchio_model.h"

namespace damotion {
namespace casadi {

PinocchioModelWrapper &PinocchioModelWrapper::operator=(
    pinocchio::Model model) {
  // Cast model to type
  model_ = model.cast<sym_t>();
  // Create data for model
  data_ = pinocchio::DataTpl<sym_t>(model_);

  // Create symbolic vectors in Casadi
  qc_ = sym_t::sym("q", model_.nq);
  vc_ = sym_t::sym("v", model_.nv);
  ac_ = sym_t::sym("a", model_.nv);
  uc_ = sym_t::sym("u", model_.nv);

  // Copy to Eigen
  toEigen(qc_, qe_);
  toEigen(vc_, ve_);
  toEigen(ac_, ae_);
  toEigen(uc_, ue_);

  return *this;
}

::casadi::Function PinocchioModelWrapper::aba() {
  // Compute forward dynamics by ABA algorithm
  sym_vec_eig_t acc_e = pinocchio::aba<sym_t>(model_, data_, qe_, ve_, ue_);
  sym_t acc_c;
  toCasadi(acc_e, acc_c);

  // Create function for ABA
  return ::casadi::Function(model_.name + "_aba", {qc_, vc_, uc_}, {acc_c},
                            {"q", "v", "u"}, {"a"});
}

::casadi::Function PinocchioModelWrapper::rnea() {
  // Convert to SX
  sym_vec_eig_t tau_e = pinocchio::rnea<sym_t>(model_, data_, qe_, ve_, ae_);
  sym_t tau_c;
  toCasadi(tau_e, tau_c);
  // Create function for RNEA
  return ::casadi::Function(model_.name + "_rnea", {qc_, vc_, ac_}, {tau_c},
                            {"q", "v", "a"}, {"u"});
};

/**
 * @brief Returns a casadi::Function that computes the centre-of-mass state
 * of the the system and its derivatives in time.
 *
 * @return casadi::Function
 */
::casadi::Function PinocchioModelWrapper::com(
    const pinocchio::ReferenceFrame &ref) {
  // Perform forward kinematics on model
  pinocchio::forwardKinematics(model_, data_, qe_, ve_, ae_);
  pinocchio::updateFramePlacements(model_, data_);

  // Compute centre of mass kinematics
  pinocchio::centerOfMass(model_, data_, qe_, ve_, ae_, false);
  // Get data for centre of mass
  sym_vec_eig_t com_e = data_.com[0], vcom_e = data_.vcom[0],
                acom_e = data_.acom[0];

  // Convert to preferred orientation
  if (ref == pinocchio::LOCAL) {
  } else if (ref == pinocchio::LOCAL_WORLD_ALIGNED) {
  }

  // Convert to casadi matrices
  sym_t com_c, vcom_c, acom_c;
  toCasadi(com_e, com_c);
  toCasadi(vcom_e, vcom_c);
  toCasadi(acom_e, acom_c);

  // Create end-effector function
  return ::casadi::Function(model_.name + "_com", {qc_, vc_, ac_},
                            {densify(com_c), densify(vcom_c), densify(acom_c)},
                            {"q", "v", "a"}, {"com", "vcom", "acom"});
}

/**
 * @brief Creates a casadi::Function that returns the state of the
 * end-effector (i.e. position, velocity and acceleration) within the
 * requested frame.
 *
 * @param frame_name
 * @param ref
 * @return casadi::Function
 */
::casadi::Function PinocchioModelWrapper::EndEffector(
    const std::string &frame_name, const pinocchio::ReferenceFrame &ref) {
  // Perform forward kinematics on model
  pinocchio::forwardKinematics(model_, data_, qe_, ve_, ae_);
  pinocchio::updateFramePlacements(model_, data_);
  // Get SE3 data for the target frame
  pinocchio::SE3Tpl<sym_t> se3_frame = data_.oMf[model_.getFrameId(frame_name)];

  // Convert to preferred orientation
  if (ref == pinocchio::LOCAL) {
  } else if (ref == pinocchio::LOCAL_WORLD_ALIGNED) {
  }

  sym_vec_eig_t pos_e(7), vel_e(6), acc_e(6);
  // Translational component
  pos_e.topRows(3) = se3_frame.translation();
  // Rotational component
  Eigen::Matrix3<sym_t> R = se3_frame.rotation();
  Eigen::Quaternion<sym_t> qR;
  pinocchio::quaternion::assignQuaternion(qR, R);

  // Convert rotation matrix to quaternion representation [w, x, y, z]
  pos_e.bottomRows(4) << qR.w(), qR.vec();

  // Compute velocity of the point at the end-effector frame with
  // respect to the chosen reference frame
  vel_e = pinocchio::getFrameVelocity(model_, data_,
                                      model_.getFrameId(frame_name), ref)
              .toVector();

  // Compute acceleration of the point at the end-effector frame with
  // respect to the chosen reference frame
  acc_e = pinocchio::getFrameClassicalAcceleration(
              model_, data_, model_.getFrameId(frame_name), ref)
              .toVector();

  // Convert to casadi matrices
  sym_t pos_c, vel_c, acc_c;
  toCasadi(pos_e, pos_c);
  toCasadi(vel_e, vel_c);
  toCasadi(acc_e, acc_c);

  // Create end-effector function
  return ::casadi::Function(model_.name + "_" + frame_name, {qc_, vc_, ac_},
                            {densify(pos_c), densify(vel_c), densify(acc_c)},
                            {"q", "v", "a"}, {"pos", "vel", "acc"});
}

// Centroidal momentum
::casadi::Function PinocchioModelWrapper::CentroidalMomentum() {
  pinocchio::computeCentroidalMomentum(model_, data_, qe_, ve_);
  // Create a six-dimensional vector for the centroidal momentum
  sym_vec_eig_t cm_e = data_.hg.toVector();
  sym_t cm_c;
  toCasadi(cm_e, cm_c);

  return ::casadi::Function(model_.name + "_centroidal_momentum",
                            {qc_, vc_, ac_}, {cm_c}, {"q", "v", "a"}, {"cm"});
}

}  // namespace casadi
}  // namespace damotion
