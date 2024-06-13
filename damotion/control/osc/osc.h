#ifndef OSC_OSC_H
#define OSC_OSC_H

#include <Eigen/Core>
#include <algorithm>
#include <casadi/casadi.hpp>
#include <map>
#include <string>

#include "damotion/casadi/codegen.h"
#include "damotion/casadi/eigen.h"
#include "damotion/casadi/function.h"
#include "damotion/casadi/pinocchio_model.h"
#include "damotion/common/math/quaternion.h"
#include "damotion/common/profiler.h"
#include "damotion/control/fwd.h"
#include "damotion/control/osc/tasks/contact.h"
#include "damotion/control/osc/tasks/motion.h"
#include "damotion/optimisation/constraints/constraints.h"
#include "damotion/optimisation/costs/costs.h"
#include "damotion/optimisation/program.h"

namespace damotion {
namespace control {
namespace osc {

// Think about not making it a solver, but instead a collection of utilities to
// create a program?

std::shared_ptr<opt::LinearConstraint<Eigen::MatrixXd>>
LinearisedFrictionConstraint();

/**
 * @brief Basic implementation of an Operational Space Controller for a system
 * with state (qpos, qvel, qacc) that undergoes contact as well as performing
 * motion tasks.
 *
 */
class OSC : public opt::Program<Eigen::MatrixXd> {
 public:
  OSC() = default;
  ~OSC() = default;

  OSC(int nq, int nv, int nu);

  /**
   * @brief Registers a motion task for the
   *
   * @param task
   * @param xpos Symbolic representation of the task \f$ x(q) \f$
   * @param xvel Symbolic representation of the task velocity \f$ \dot{x}(q) \f$
   * @param xacc Symbolic representation of the task acceleration \f$
   * \ddot{x}(q) \f$
   * @return void
   */
  void AddMotionTask(MotionTask::SharedPtr &task, const casadi::SX &xpos,
                     const casadi::SX &xvel, const casadi::SX &xacc);

  /**
   * @brief Add a contact point to the OSC.
   *
   * @param task
   * @param xpos Symbolic representation of the task \f$ x(q) \f$
   * @param xvel Symbolic representation of the task velocity \f$ \dot{x}(q) \f$
   * @param xacc Symbolic representation of the task acceleration \f$
   * \ddot{x}(q) \f$
   * @return void
   */
  void AddContactTask(ContactTask::SharedPtr &task, const casadi::SX &xpos,
                      const casadi::SX &xvel, const casadi::SX &xacc);

  /**
   * @brief For a constraint that can be written in the form h(q) = 0,
   * computes the linear constraint imposed by the second derivative in time
   * of the constraint.
   *
   * @param name
   * @param c The holonomic constraint c(q) = 0
   * @param dcdt The first derivative of c(q) wrt time
   * @param d2cdt2 The second derivative of c(q) wrt time
   */
  void AddHolonomicConstraint(const std::string &name, const casadi::SX &c,
                              const casadi::SX &dcdt, const casadi::SX &d2cdt2);

  /**
   * @brief Returns the collective joint-space forces created by the
   * equality-constraints currently within the program.
   *
   * @return const casadi::SX
   */
  casadi::SX GetJointSpaceConstraintForces() {
    return joint_space_constraint_forces_;
  }

  /**
   * @brief Update the program given changes to the configuration and velocity
   * of the system.
   *
   * @param qpos
   * @param qvel
   */
  void Update(const Eigen::VectorXd &qpos, const Eigen::VectorXd &qvel) {
    // TODO Store these references
    GetParameterRef(qpos_) << qpos;
    GetParameterRef(qvel_) << qvel;

    // Update each task given the updated data and references
    for (size_t i = 0; i < motion_tasks_.size(); ++i) {
      // Set desired task accelerations
      motion_tasks_[i]->ComputeMotionError();
      motion_task_parameters_[i].xaccd << -motion_tasks_[i]->GetPDError();
    }
    for (size_t i = 0; i < contact_tasks_.size(); ++i) {
      // Set desired task accelerations
      contact_tasks_[i]->ComputeMotionError();
      contact_task_parameters_[i].xaccd << -contact_tasks_[i]->GetPDError();
    }
  }

  /**
   * @brief Updates the contact data within the program for the contact task
   * task
   *
   * @param name
   * @param task
   */
  void UpdateContactPoint(const ContactTask::SharedPtr &task) {
    // TODO Get task index
    int task_idx = 0;
    if (task->inContact) {
      // Get parameters related to the contact task
      ContactTaskParameters &p = contact_task_parameters_[task_idx];
      // Update bounding box constraints
      contact_force_bounds_[task_idx].Get().setBounds(task->fmin(),
                                                      task->fmax());
      p.mu << task->mu();
      p.normal << task->normal();

    } else {
      contact_force_bounds_[task_idx].Get().setBounds(
          opt::BoundsType::kEquality);
    }
  }

  /**
   * @brief Set the Generalised Acceleration Variables object by providing
   * symbolic and optimisation variables to the program.
   *
   * @param var
   * @param sym
   */
  void SetGeneralisedAccelerationVariables(const sym::VariableVector &var,
                                           const casadi::SX &sym) {
    assert(var.size() == nv_ && sym.rows() == nv_ &&
           "Generalised acceleration vectors must be size nv");
    qacc_ = var;
    qacc_sx_ = sym;
    AddDecisionVariables(var);
  }
  void SetControlVariables(const sym::VariableVector &var,
                           const casadi::SX &sym) {
    assert(var.size() == nu_ && sym.rows() == nu_ &&
           "Control vectors must be size nu");
    ctrl_ = var;
    ctrl_sx_ = sym;
    AddDecisionVariables(var);
  }

  void AddContactForceVariables(const sym::VariableVector &var,
                                const casadi::SX &sym) {
    lambda_.push_back(var);
    lambda_sx_.push_back(sym);
    AddDecisionVariables(var);
  }

  void AddConstraintForceVariables(const sym::VariableVector &var,
                                   const casadi::SX &sym) {
    lambda_.push_back(var);
    lambda_sx_.push_back(sym);
    AddDecisionVariables(var);
  }

  std::vector<sym::VariableVector> &ConstraintForceVariables() {
    return lambda_;
  }

  std::vector<casadi::SX> &ConstraintForceVariablesSymbolic() {
    return lambda_sx_;
  }

  void AddGeneralisedPositionParameters(const sym::ParameterVector &par,
                                        const casadi::SX &sym) {
    assert(par.size() == nq_ && sym.rows() == nq_ &&
           "Generalised position vectors must be size nq");
    qpos_sx_ = sym;
    AddParameters(par);
  }

  void AddGeneralisedVelocityParameters(const sym::ParameterVector &par,
                                        const casadi::SX &sym) {
    assert(par.size() == nv_ && sym.rows() == nv_ &&
           "Generalised velocity vectors must be size nv");
    qvel_sx_ = sym;
    AddParameters(par);
  }

  /* Optimisation optimisation variables */

  sym::VariableVector qacc_;
  sym::VariableVector ctrl_;
  std::vector<sym::VariableVector> lambda_ = {};

  sym::ParameterVector qpos_;
  sym::ParameterVector qvel_;

  /* Casadi symbolic variables */

  casadi::SX qacc_sx_;
  casadi::SX ctrl_sx_;
  casadi::SXVector lambda_sx_ = {};

  casadi::SX qpos_sx_;
  casadi::SX qvel_sx_;

 private:
  int nq_ = 0;
  int nv_ = 0;
  int nu_ = 0;

  // Vector of motion tasks and parameter references for the program
  std::vector<MotionTask::SharedPtr> motion_tasks_;
  struct MotionTaskParameters {
    MotionTaskParameters(const Eigen::Map<Eigen::VectorXd> &xaccd,
                         const Eigen::Map<Eigen::VectorXd> &w)
        : xaccd(xaccd), w(w) {}
    Eigen::Map<Eigen::VectorXd> xaccd;
    Eigen::Map<Eigen::VectorXd> w;
  };
  std::vector<MotionTaskParameters> motion_task_parameters_;

  // Vector of contact tasks and parameter references for the program
  std::vector<ContactTask::SharedPtr> contact_tasks_;
  // Parameters for each contact point
  struct ContactTaskParameters {
    ContactTaskParameters(const Eigen::Map<Eigen::VectorXd> &xaccd,
                          const Eigen::Map<Eigen::VectorXd> &w,
                          const Eigen::Map<Eigen::VectorXd> &mu,
                          const Eigen::Map<Eigen::VectorXd> &normal)
        : xaccd(xaccd), w(w), mu(mu), normal(normal) {}

    Eigen::Map<Eigen::VectorXd> xaccd;
    Eigen::Map<Eigen::VectorXd> w;
    Eigen::Map<Eigen::VectorXd> mu;
    Eigen::Map<Eigen::VectorXd> normal;
  };
  std::vector<ContactTaskParameters> contact_task_parameters_;

  // Bounding box constraints for contact forces
  std::vector<opt::Binding<opt::BoundingBoxConstraint<Eigen::MatrixXd>>>
      contact_force_bounds_;

  // Friction cone constraint
  std::shared_ptr<opt::LinearConstraint<Eigen::MatrixXd>> friction_cone_con_;

  // Constrained dynamics
  casadi::SX joint_space_constraint_forces_;
};

}  // namespace osc
}  // namespace control
}  // namespace damotion

#endif /* OSC_OSC_H */
