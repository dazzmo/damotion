#ifndef UTILS_PINOCCHIO_MODEL_H
#define UTILS_PINOCCHIO_MODEL_H

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/autodiff/casadi/math/quaternion.hpp>
#include <pinocchio/autodiff/casadi/utils/static-if.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "damotion/model/frame.h"
#include "damotion/utils/eigen_wrapper.h"
#include "damotion/utils/log.h"

namespace damotion {
namespace utils {
namespace casadi {

class PinocchioModelWrapper {
 public:
  // Define casadi autodiff type
  using AD = ::casadi::SXElem;

  PinocchioModelWrapper() = default;
  PinocchioModelWrapper(pinocchio::Model &model) { *this = model; }

  PinocchioModelWrapper &operator=(pinocchio::Model model);

  /**
   * @brief End effector data for a model
   *
   */
  class EndEffector : public model::TargetFrame {
   public:
    typedef int Id;
    /**
     * @brief Unique ID of the end-effector
     *
     * @return const Id&
     */
    const Id &id() const { return id_; }

    /**
     * @brief Construct a new End Effector object using a frame within the
     * provided model.
     *
     * @param name
     * @param model
     * @param data
     */
    EndEffector(
        const std::string &frame_name,
        pinocchio::ModelTpl<::casadi::Matrix<AD>> &model,
        pinocchio::DataTpl<::casadi::Matrix<AD>> &data,
        const pinocchio::ReferenceFrame &ref = pinocchio::LOCAL_WORLD_ALIGNED)
        : name_(frame_name) {
      typedef ::casadi::Matrix<AD> MatrixType;

      // Create the function for the end-effector
      MatrixType qpos = MatrixType::sym("q", model.nq),
                 qvel = MatrixType::sym("v", model.nv),
                 qacc = MatrixType::sym("a", model.nv);

      // Eigen-equivalents
      Eigen::VectorX<MatrixType> qpos_e, qvel_e, qacc_e;
      toEigen(qpos, qpos_e);
      toEigen(qvel, qvel_e);
      toEigen(qacc, qacc_e);

      // Perform forward kinematics on model
      pinocchio::forwardKinematics(model, data, qpos_e, qvel_e, qacc_e);
      pinocchio::updateFramePlacements(model, data);
      // Get SE3 data for the target frame
      pinocchio::SE3Tpl<MatrixType> se3_frame =
          data.oMf[model.getFrameId(frame_name)];

      // Convert to preferred orientation
      if (ref == pinocchio::LOCAL) {
      } else if (ref == pinocchio::LOCAL_WORLD_ALIGNED) {
      }

      Eigen::VectorX<MatrixType> pos_e(7), vel_e(6), acc_e(6);

      // Translational component
      pos_e.topRows(3) = se3_frame.translation();

      // Rotational component
      Eigen::Matrix3<MatrixType> R = se3_frame.rotation();
      Eigen::Quaternion<MatrixType> qR;
      pinocchio::quaternion::assignQuaternion(qR, R);

      // Convert rotation matrix to quaternion representation
      pos_e.bottomRows(4) << qR.w(), qR.vec();

      // Compute velocity of the point at the end-effector frame with
      // respect to the chosen reference frame
      vel_e = pinocchio::getFrameVelocity(model, data,
                                          model.getFrameId(frame_name), ref)
                  .toVector();

      // Compute acceleration of the point at the end-effector frame with
      // respect to the chosen reference frame
      acc_e = pinocchio::getFrameClassicalAcceleration(
                  model, data, model.getFrameId(frame_name), ref)
                  .toVector();

      // Convert to casadi matrices
      MatrixType pos, vel, acc;
      toCasadi(pos_e, pos);
      toCasadi(vel_e, vel);
      toCasadi(acc_e, acc);

      // Create end-effector function
      SetFunction(
          ::casadi::Function(model.name + "_" + frame_name, {qpos, qvel, qacc},
                             {densify(pos), densify(vel), densify(acc)},
                             {"qpos", "qvel", "qacc"}, {"pos", "vel", "acc"}));

      id_ = CreateId();
    }

    ~EndEffector() = default;

    const std::string &name() const { return name_; }

   private:
    // Dimension of the end-effector
    int dim_ = 0;
    // Name of the end-effector
    std::string name_;

    Id id_;

    Id CreateId() {
      static Id next_id = 0;
      Id id = next_id++;
      return id;
    }
  };

  /**
   * @brief Centre of mass data for a model
   *
   */
  class CentreOfMass : public model::TargetFrame {
   public:
    /**
     * @brief Construct a new Centre of Mass object using a frame within the
     * provided model.
     *
     * @param name
     * @param model
     * @param data
     */
    CentreOfMass(pinocchio::ModelTpl<::casadi::Matrix<AD>> &model,
                 pinocchio::DataTpl<::casadi::Matrix<AD>> &data,
                 const pinocchio::ReferenceFrame &ref = pinocchio::WORLD)
        : name_("centre_of_mass") {
      typedef ::casadi::Matrix<AD> MatrixType;

      // Create the function for the end-effector
      MatrixType qpos = MatrixType::sym("q", model.nq),
                 qvel = MatrixType::sym("v", model.nv),
                 qacc = MatrixType::sym("a", model.nv);

      // Eigen-equivalents
      Eigen::VectorX<MatrixType> qpos_e, qvel_e, qacc_e;
      toEigen(qpos, qpos_e);
      toEigen(qvel, qvel_e);
      toEigen(qacc, qacc_e);

      // Perform forward kinematics on model
      pinocchio::forwardKinematics(model, data, qpos_e, qvel_e, qacc_e);
      pinocchio::updateFramePlacements(model, data);

      // Compute centre of mass kinematics
      pinocchio::centerOfMass(model, data, qpos_e, qvel_e, qacc_e, false);
      // Get data for centre of mass
      Eigen::Vector<MatrixType, 3> com = data.com[0], vcom = data.vcom[0],
                                   acom = data.acom[0];

      // Convert to preferred orientation
      if (ref == pinocchio::LOCAL) {
      } else if (ref == pinocchio::LOCAL_WORLD_ALIGNED) {
      }

      // Convert to casadi matrices
      MatrixType com_sym, vcom_sym, acom_sym;
      toCasadi(com, com_sym);
      toCasadi(vcom, vcom_sym);
      toCasadi(acom, acom_sym);

      // Create end-effector function
      SetFunction(::casadi::Function(
          model.name + "_com", {qpos, qvel, qacc},
          {densify(com_sym), densify(vcom_sym), densify(acom_sym)},
          {"qpos", "qvel", "qacc"}, {"com", "vcom", "acom"}));
    }

    ~CentreOfMass() = default;

    const std::string &name() const { return name_; }

   private:
    // Dimension of the end-effector
    int dim_ = 0;
    // Name of the end-effector
    std::string name_;
  };

  /**
   * @brief Returns a casadi::Function that computes the forward-dynamics of
   * the system through the Articulated Body Algorithm (ABA)
   *
   * @return casadi::Function
   */
  ::casadi::Function aba();

  /**
   * @brief Returns a casadi::Function that computes the inverse-dynamics of
   * the system through the Recursive Newton-Euler Algorithm (ABA)
   *
   * @return casadi::Function
   */
  ::casadi::Function rnea();

  /**
   * @brief Returns a casadi::Function that computes the centre-of-mass state
   * of the the system and its derivatives in time.
   *
   * @return casadi::Function
   */
  std::shared_ptr<CentreOfMass> com() {
    return std::make_shared<CentreOfMass>(model_, data_);
  }

  std::shared_ptr<EndEffector> AddEndEffector(const std::string &name) {
    auto ee = std::make_shared<EndEffector>(name, model_, data_);
    // Add end effector to the vector
    ee_.push_back(ee);
    return ee;
  }

  pinocchio::ModelTpl<::casadi::Matrix<AD>> &model() { return model_; }
  pinocchio::DataTpl<::casadi::Matrix<AD>> &data() { return data_; }

 private:
  pinocchio::ModelTpl<::casadi::Matrix<AD>> model_;
  pinocchio::DataTpl<::casadi::Matrix<AD>> data_;

  std::vector<std::shared_ptr<EndEffector>> ee_;
  std::unordered_map<std::string, int> ee_idx_;
};

}  // namespace casadi
}  // namespace utils
}  // namespace damotion

#endif /* UTILS_PINOCCHIO_MODEL_H */
