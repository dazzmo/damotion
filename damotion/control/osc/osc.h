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
#include "damotion/utils/casadi.h"
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
  void AddContactPoint(const std::shared_ptr<ContactTask> &task);

  /**
   * @brief Adds a generic motion task to the OSC.
   *
   * @param task
   * @return void
   */
  void AddMotionTask(const std::shared_ptr<MotionTask> &task);

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
  void AddUnconstrainedInverseDynamics(
      const casadi::SX &dynamics, const casadi::SXVector &parameters_sym = {},
      const sym::ParameterVector &parameters = {}) {
    assert(parameters_sym.size() == parameters.size() &&
           "Parameter inputs must be same size!");
    // Initialise expression
    constrained_dynamics_ = dynamics;
    // Initialise inputs to the system
    constrained_dynamics_sym_ =
        casadi::SX::vertcat({symbolic_terms_->qacc(), symbolic_terms_->ctrl()});
    constrained_dynamics_var_.resize(nv_ + nu_);
    constrained_dynamics_var_ << variables_->qacc(), variables_->ctrl();

    // Initialise parameters
    constrained_dynamics_par_sym_ = {symbolic_terms_->qpos(),
                                     symbolic_terms_->qvel()};
    for (auto &pi : parameters_sym) {
      constrained_dynamics_par_sym_.push_back(pi);
    }
    constrained_dynamics_par_.push_back(qpos_param_);
    constrained_dynamics_par_.push_back(qvel_param_);
    for (auto &pi : parameters) {
      constrained_dynamics_par_.push_back(pi);
    }
  }

  /**
   * @brief Creates the program after adding all necessary tasks, constraints
   * and objectives
   *
   */
  void CreateProgram() {
    // Create linear constraint for the constrained dynamics
    casadi::SX A, b;
    casadi::SX::linear_coeff(constrained_dynamics_, constrained_dynamics_sym_,
                             A, b, true);
    auto con = std::make_shared<opt::LinearConstraint<Eigen::MatrixXd>>(
        "dynamics", A, b, constrained_dynamics_par_sym_,
        opt::BoundsType::kEquality);
    // Add constraint to program
    program_.AddLinearConstraint(con, {constrained_dynamics_var_},
                                 constrained_dynamics_par_);

    // Construct the decision variable vector  [qacc, ctrl, lambda]
    sym::VariableVector x(program_.NumberOfDecisionVariables());
    // Set it as qacc, ctrl, constraint forces
    x.topRows(nv_ + nu_) << variables_->qacc(), variables_->ctrl();
    // Add any constraint forces
    int idx = nv_ + nu_;
    for (int i = 0; i < variables_->lambda().size(); ++i) {
      x.middleRows(idx, variables_->lambda(i).size()) = variables_->lambda(i);
      idx += variables_->lambda(i).size();
    }
    program_.SetDecisionVariableVector(x);
  }

  void Update(const Eigen::VectorXd &qpos, const Eigen::VectorXd &qvel) {
    program_.SetParameterValues(qpos_param_, qpos);
    program_.SetParameterValues(qvel_param_, qvel);

    // Update each task given the updated data and references
    for (int i = 0; i < motion_tasks_.size(); ++i) {
      // Set desired task accelerations
      motion_tasks_[i]->ComputeMotionError();
      GetProgram().SetParameterValues(motion_task_parameters_[i].xaccd,
                                      -motion_tasks_[i]->GetPDError());
    }

    for (int i = 0; i < contact_tasks_.size(); ++i) {
      // Compute the pose error for their costs
      contact_tasks_[i]->ComputeMotionError();
      GetProgram().SetParameterValues(contact_task_parameters_[i].xaccd,
                                      -contact_tasks_[i]->GetPDError());
    }

    // Program is updated
  }

  /**
   * @brief Updates the contact data within the program for the contact task
   * task
   *
   * @param name
   * @param task
   */
  void UpdateContactPoint(const std::shared_ptr<ContactTask> &task) {
    int task_idx = 0;
    if (task->inContact) {
      // Get parameters related to the contact task
      ContactTaskParameters &p = contact_task_parameters_[task_idx];
      // Set in contact
      contact_force_bounds_[task_idx].Get().SetBounds(task->fmin(),
                                                      task->fmax());

      program_.SetParameterValues(p.mu, Eigen::Vector<double, 1>(task->mu()));
      program_.SetParameterValues(p.normal, task->normal());
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

  class Variables {
   public:
    Variables() = default;
    ~Variables() = default;

    Variables(int nq, int nv, int nu) {
      qacc_ = sym::CreateVariableVector("qacc", nv),
      ctrl_ = sym::CreateVariableVector("ctrl", nu);
      lambda_ = {};
    }

    /**
     * @brief Generalised acceleration variables for the program (nv x 1)
     *
     * @return const sym::VariableVector&
     */
    const sym::VariableVector &qacc() const { return qacc_; }
    sym::VariableVector &qacc() { return qacc_; }

    /**
     * @brief Control input variables for the program (nu x 1)
     *
     * @return const sym::VariableVector&
     */
    const sym::VariableVector &ctrl() const { return ctrl_; }
    sym::VariableVector &ctrl() { return ctrl_; }

    /**
     * @brief Constraint force variable vector
     *
     * @return const std::vector<sym::VariableVector>&
     */
    const std::vector<sym::VariableVector> &lambda() const { return lambda_; }

    const sym::VariableVector &lambda(const int i) const { return lambda_[i]; }

    /**
     * @brief Adds constraint force variables to the variable list
     *
     * @param lambda
     */
    void AddConstraintForces(const sym::VariableVector &lambda) {
      lambda_.push_back(lambda);
    }

   private:
    sym::VariableVector qacc_;
    sym::VariableVector ctrl_;
    std::vector<sym::VariableVector> lambda_;
  };

  const Variables &GetVariables() const { return *variables_; }
  Variables &GetVariables() { return *variables_; }

  class SymbolicTerms {
   public:
    SymbolicTerms() = default;
    ~SymbolicTerms() = default;

    SymbolicTerms(int nq, int nv, int nu) {
      qpos_ = casadi::SX::sym("qpos", nq), qvel_ = casadi::SX::sym("qvel", nv),
      qacc_ = casadi::SX::sym("qacc", nv), ctrl_ = casadi::SX::sym("ctrl", nu);
      lambda_ = {};
    }

    const casadi::SX &qpos() const { return qpos_; }
    const casadi::SX &qvel() const { return qvel_; }
    const casadi::SX &qacc() const { return qacc_; }
    const casadi::SX &ctrl() const { return ctrl_; }

    const std::vector<casadi::SX> &lambda() const { return lambda_; }

    const casadi::SX &lambda(const int i) const { return lambda_[i]; }

    void AddConstraintForces(const casadi::SX &lambda) {
      lambda_.push_back(lambda);
    }

   private:
    casadi::SX qpos_;
    casadi::SX qvel_;
    casadi::SX qacc_;
    casadi::SX ctrl_;
    std::vector<casadi::SX> lambda_;
  };

  const SymbolicTerms &GetSymbolicTerms() const { return *symbolic_terms_; }
  SymbolicTerms &GetSymbolicTerms() { return *symbolic_terms_; }

 private:
  int nq_ = 0;
  int nv_ = 0;
  int nu_ = 0;

  void AddConstraintsToDynamics(const casadi::SX &lambda,
                                const sym::VariableVector &lambda_var,
                                const casadi::SX &J,
                                const casadi::SXVector &parameters,
                                const sym::ParameterVector &parameters_var) {
    // Add variables to the input vectors
    constrained_dynamics_sym_ =
        casadi::SX::vertcat({constrained_dynamics_sym_, lambda});
    constrained_dynamics_var_.conservativeResize(
        constrained_dynamics_var_.size() + lambda_var.size());
    constrained_dynamics_var_.bottomRows(lambda_var.size()) << lambda_var;
    for (auto &pi : parameters) {
      constrained_dynamics_par_sym_.push_back(pi);
    }
    for (auto &pi : parameters_var) {
      constrained_dynamics_par_.push_back(pi);
    }
    // Add to the constraint
    constrained_dynamics_ -= mtimes(J.T(), lambda);
  }

  // Conventional optimisation variables
  std::unique_ptr<Variables> variables_;
  // Symbolic variables used for constraint/objective generation
  std::unique_ptr<SymbolicTerms> symbolic_terms_;

  // Default parameters
  sym::Parameter qpos_param_;
  sym::Parameter qvel_param_;

  // Vector of motion tasks and parameter references for the program
  std::vector<std::shared_ptr<MotionTask>> motion_tasks_;
  struct MotionTaskParameters {
    sym::Parameter xaccd;
    sym::Parameter w;
  };
  std::vector<MotionTaskParameters> motion_task_parameters_;

  // Vector of contact tasks and parameter references for the program
  std::vector<std::shared_ptr<ContactTask>> contact_tasks_;
  // Parameters for each contact point
  struct ContactTaskParameters {
    sym::Parameter xaccd;
    sym::Parameter w;
    sym::Parameter mu;
    sym::Parameter normal;
  };
  std::vector<ContactTaskParameters> contact_task_parameters_;
  std::vector<opt::Binding<opt::BoundingBoxConstraint<Eigen::MatrixXd>>>
      contact_force_bounds_;

  // Friction cone constraint
  std::shared_ptr<opt::LinearConstraint<Eigen::MatrixXd>> friction_cone_con_;

  // Constrained dynamics
  casadi::SX constrained_dynamics_sym_;
  casadi::SXVector constrained_dynamics_par_sym_;
  sym::VariableVector constrained_dynamics_var_;
  sym::ParameterVector constrained_dynamics_par_;
  sym::Expression constrained_dynamics_;

  // Underlying program for the OSC
  opt::Program program_;
};

}  // namespace osc
}  // namespace control
}  // namespace damotion

#endif /* OSC_OSC_H */
