/**
 * @file pinocchio_model.h
 * @author Damian Abood (damian.abood@sydney.edu.au)
 * @brief Wrapper class for Pinocchio's Model class to be used within a Casadi
 * context, providing casadi::Function objects for symbolic evaluation of
 * conventional kinodynamic quantities.
 * @version 0.1
 * @date 2024-05-09
 *
 *
 */
#ifndef CASADI_PINOCCHIO_MODEL_H
#define CASADI_PINOCCHIO_MODEL_H

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
// Centroidal momentum
#include <pinocchio/algorithm/centroidal.hpp>

#include "damotion/casadi/eigen.h"
#include "damotion/core/math/log.h"

namespace damotion {
namespace casadi {

class PinocchioModelWrapper {
 public:
  // Define casadi autodiff type
  using sym_t = ::casadi::SX;
  using sym_vec_eig_t = Eigen::VectorX<sym_t>;

  PinocchioModelWrapper() = default;
  PinocchioModelWrapper(pinocchio::Model &model) { *this = model; }

  PinocchioModelWrapper &operator=(pinocchio::Model model);

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
  ::casadi::Function com(
      const pinocchio::ReferenceFrame &ref = pinocchio::WORLD);
  /**
   * @brief Creates a casadi::Function that returns the state of the
   * end-effector (i.e. position, velocity and acceleration) within the
   * requested frame.
   *
   * @param frame_name
   * @param ref
   * @return casadi::Function
   */
  ::casadi::Function EndEffector(
      const std::string &frame_name,
      const pinocchio::ReferenceFrame &ref = pinocchio::LOCAL_WORLD_ALIGNED);

  /**
   * @brief Returns the six-dimensional centroidal momentum vector of the model
   * about its centre of mass.
   *
   * @return ::casadi::Function
   */
  ::casadi::Function CentroidalMomentum();

  pinocchio::ModelTpl<sym_t> &model() { return model_; }
  pinocchio::DataTpl<sym_t> &data() { return data_; }

 private:
  // Casadi symbolic representation of generalised configuration
  sym_t qc_;
  // Casadi symbolic representation of generalised velocity
  sym_t vc_;
  // Casadi symbolic representation of generalised acceleration
  sym_t ac_;
  // Casadi symbolic representation of generalised input
  sym_t uc_;

  // Eigen symbolic representation of generalised configuration
  Eigen::VectorX<sym_t> qe_;
  // Eigen symbolic representation of generalised velocity
  Eigen::VectorX<sym_t> ve_;
  // Eigen symbolic representation of generalised acceleration
  Eigen::VectorX<sym_t> ae_;
  // Eigen symbolic representation of generalised input
  Eigen::VectorX<sym_t> ue_;

  // Symbolic Pinocchio model class
  pinocchio::ModelTpl<sym_t> model_;
  // Symbolic Pinocchio model data class
  pinocchio::DataTpl<sym_t> data_;
};

}  // namespace casadi
}  // namespace damotion

#endif/* CASADI_PINOCCHIO_MODEL_H */
