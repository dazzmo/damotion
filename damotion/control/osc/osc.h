#ifndef OSC_OSC_H
#define OSC_OSC_H

#include <Eigen/Core>
#include <algorithm>
#include <casadi/casadi.hpp>
#include <map>
#include <string>

#include "damotion/common/profiler.h"
#include "damotion/control/osc/tasks/contact.h"
#include "damotion/control/osc/tasks/motion.h"
#include "damotion/optimisation/constraints/constraints.h"
#include "damotion/optimisation/costs/costs.h"
#include "damotion/optimisation/program.h"
#include "damotion/utils/codegen.h"
#include "damotion/utils/eigen_wrapper.h"
#include "damotion/utils/pinocchio_model.h"

namespace opt = damotion::optimisation;
namespace sym = damotion::symbolic;
namespace utils = damotion::utils;

namespace damotion {
namespace control {
namespace osc {

typedef model::TargetFrame TargetFrame;

std::shared_ptr<opt::LinearConstraint<Eigen::MatrixXd>>
LinearisedFrictionConstraint();

// TODO - Place this in a utility
Eigen::Quaternion<double> RPYToQuaterion(const double roll, const double pitch,
                                         const double yaw);

/**
 * @brief Basic implementation of an Operational Space Controller for a system
 * with state (qpos, qvel, qacc) that undergoes contact as well as performing
 * motion tasks. For anything more sophisticated such as including other
 * parameters or objectives, this controller can be modified explicitly by
 * accessing its underlying program through GetProgram(). If that is not enough,
 * we encourage creating a custom controller using the Program class.
 *
 */
class OSC {
 public:
  OSC() = default;
  ~OSC() = default;

  OSC(int nq, int nv, int nu);

  /**
   * @brief Add a contact point to the OSC.
   *
   * @param task
   * @return void
   */
  void AddContactTask(std::shared_ptr<ContactTask> &task,
                      model::symbolic::TargetFrame &frame);

  /**
   * @brief Registers a motion task for the
   *
   * @param task
   * @return void
   */
  void AddMotionTask(std::shared_ptr<MotionTask> &task,
                     model::symbolic::TargetFrame &frame);

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
   * @brief Sets the unconstrained inverse dynamics for the OSC program. Note
   * that this must be called before adding any other tasks so that constraint
   * forces can be added to this expression.
   *
   * @param dynamics
   */
  void AddUnconstrainedInverseDynamics(const casadi::SX &dynamics) {
    // Initialise expression
    constrained_dynamics_ = dynamics;
  }

  /**
   * @brief Creates the program after adding all necessary tasks, constraints
   * and objectives
   *
   */
  void CreateProgram() {
    // Create linear constraint for the constrained dynamics
    casadi::SX A, b;
    // TODO Create single vector
    casadi::SX x_dyn = casadi::SX::vertcat(constrained_dynamics_.Variables());
    casadi::SX::linear_coeff(constrained_dynamics_, x_dyn, A, b, true);
    auto con = std::make_shared<opt::LinearConstraint<Eigen::MatrixXd>>(
        "dynamics", A, b, constrained_dynamics_.Parameters(),
        opt::BoundsType::kEquality);
    // Add constraint to program
    program_.AddLinearConstraint(con, {qacc_, ctrl_}, {qpos_, qvel_});

    // Construct the decision variable vector  [qacc, ctrl, lambda]
    sym::VariableVector x(program_.NumberOfDecisionVariables());
    // Set it as qacc, ctrl, constraint forces
    x.topRows(nv_ + nu_) << qacc_, ctrl_;
    // Add any constraint forces
    int idx = nv_ + nu_;
    for (size_t i = 0; i < lambda_.size(); ++i) {
      x.middleRows(idx, lambda_[i].size()) = lambda_[i];
      idx += lambda_[i].size();
    }
    program_.SetDecisionVariableVector(x);
  }

  void Update(const Eigen::VectorXd &qpos, const Eigen::VectorXd &qvel) {
    // TODO Store these references
    program_.GetParameterRef(qpos_) << qpos;
    program_.GetParameterRef(qvel_) << qvel;

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
  void UpdateContactPoint(const std::shared_ptr<ContactTask> &task) {
    // TODO Get task index
    int task_idx = 0;
    if (task->inContact) {
      // Get parameters related to the contact task
      ContactTaskParameters &p = contact_task_parameters_[task_idx];
      // Set in contact
      contact_force_bounds_[task_idx].Get().SetBounds(task->fmin(),
                                                      task->fmax());
      p.mu << task->mu();
      p.normal << task->normal();

    } else {
      contact_force_bounds_[task_idx].Get().SetBounds(
          opt::BoundsType::kEquality);
    }
  }

  /**
   * @brief Get the underlying program for the OSC. Allows you to add
   * additional variables, parameters, costs, constraints that are not
   * included in conventional OSC programs.
   *
   */
  opt::Program &GetProgram() { return program_; }

  // TODO - Set variables and parameter functions
  void SetGeneralisedAccelerationVariables(const sym::VariableVector &var,
                                           const casadi::SX &sym) {
    assert(var.size() == nv_ && sym.rows() == nv_ &&
           "Generalised acceleration vectors must be size nv");
    qacc_ = var;
    qacc_sx_ = sym;
    program_.AddDecisionVariables(var);
  }
  void SetControlVariables(const sym::VariableVector &var,
                           const casadi::SX &sym) {
    assert(var.size() == nv_ && sym.rows() == nv_ &&
           "Control vectors must be size nu");
    ctrl_ = var;
    ctrl_sx_ = sym;
    program_.AddDecisionVariables(var);
  }

  void AddContactForceVariables(const sym::VariableVector &var,
                                const casadi::SX &sym) {
    lambda_.push_back(var);
    lambda_sx_.push_back(sym);
    program_.AddDecisionVariables(var);
  }

  void AddConstraintForceVariables(const sym::VariableVector &var,
                                   const casadi::SX &sym) {
    lambda_.push_back(var);
    lambda_sx_.push_back(sym);
    program_.AddDecisionVariables(var);
  }

  void AddGeneralisedPositionParameters(const sym::ParameterVector &par,
                                        const casadi::SX &sym) {
    assert(par.size() == nq_ && sym.rows() == nq_ &&
           "Generalised position vectors must be size nq");
    qpos_sx_ = sym;
    program_.AddParameters(par);
  }

  void AddGeneralisedVelocityParameters(const sym::ParameterVector &par,
                                        const casadi::SX &sym) {
    assert(par.size() == nv_ && sym.rows() == nv_ &&
           "Generalised velocity vectors must be size nv");
    qvel_sx_ = sym;
    program_.AddParameters(par);
  }

  /* Optimisation optimisation variables */

  sym::VariableVector qacc_;
  sym::VariableVector ctrl_;
  std::vector<sym::VariableVector> lambda_;

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
  std::vector<std::shared_ptr<MotionTask>> motion_tasks_;
  struct MotionTaskParameters {
    MotionTaskParameters(const Eigen::Map<Eigen::VectorXd> &xaccd,
                         const Eigen::Map<Eigen::VectorXd> &w)
        : xaccd(xaccd), w(w) {}
    Eigen::Map<Eigen::VectorXd> xaccd;
    Eigen::Map<Eigen::VectorXd> w;
  };
  std::vector<MotionTaskParameters> motion_task_parameters_;

  // Vector of contact tasks and parameter references for the program
  std::vector<std::shared_ptr<ContactTask>> contact_tasks_;
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
  std::vector<opt::Binding<opt::BoundingBoxConstraint<Eigen::MatrixXd>>>
      contact_force_bounds_;

  // Friction cone constraint
  std::shared_ptr<opt::LinearConstraint<Eigen::MatrixXd>> friction_cone_con_;

  // Constrained dynamics
  sym::Expression constrained_dynamics_;

  // Underlying program for the OSC
  opt::Program program_;
};

}  // namespace osc
}  // namespace control
}  // namespace damotion

#endif /* OSC_OSC_H */
