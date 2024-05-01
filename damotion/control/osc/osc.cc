#include "damotion/control/osc/osc.h"

namespace damotion {
namespace control {
namespace osc {

Eigen::Quaterniond RPYToQuaterion(const double roll, const double pitch,
                                  const double yaw) {
  Eigen::AngleAxisd r = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()),
                    p = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()),
                    y = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
  return Eigen::Quaterniond(r * p * y);
}

std::shared_ptr<opt::LinearConstraint> LinearisedFrictionConstraint() {
  // Square pyramid approximation
  casadi::SX lambda = casadi::SX::sym("lambda", 3);
  casadi::SX normal = casadi::SX::sym("normal", 3);
  casadi::SX mu = casadi::SX::sym("mu");

  casadi::SX l_x = lambda(0), l_y = lambda(1), l_z = lambda(2);
  casadi::SX n_x = normal(0), n_y = normal(1), n_z = normal(2);

  // Friction cone constraint with square pyramid approximation
  sym::Expression cone;
  cone = casadi::SX(4, 1);
  cone(0) = sqrt(2.0) * l_x + mu * l_z;
  cone(1) = -sqrt(2.0) * l_x + mu * l_z;
  cone(2) = sqrt(2.0) * l_y + mu * l_z;
  cone(3) = -sqrt(2.0) * l_y + mu * l_z;
  cone.SetInputs({lambda}, {normal, mu});

  return std::make_shared<opt::LinearConstraint>("friction_cone", cone,
                                                 opt::BoundsType::kPositive);
}

OSC::OSC(int nq, int nv, int nu) : nq_(nq), nv_(nv), nu_(nu) {
  // Create default symbolic terms
  symbolic_terms_ = std::make_unique<SymbolicTerms>(nq, nv, nu);
  // Create program variables
  variables_ = std::make_unique<Variables>(nq, nv, nu);

  // Create mathematical program
  program_ = opt::Program("osc");
  program_.AddDecisionVariables(variables_->qacc());
  program_.AddDecisionVariables(variables_->ctrl());

  // Create parameters
  program_.AddParameters("qpos", nq);
  program_.AddParameters("qvel", nv);

  // Create friction cone constraint to be bounded to by contact tasks
  friction_cone_con_ = LinearisedFrictionConstraint();
}

void OSC::AddMotionTask(const std::shared_ptr<MotionTask> &task) {
  // Create objective for this task

  // Create parameter for program for task-acceleration error
  // minimisation
  Eigen::Ref<Eigen::MatrixXd> xaccd =
      program_.AddParameters(task->name() + "_xaccd", task->dim());
  // Add weighting for each task
  Eigen::Ref<Eigen::MatrixXd> w =
      program_.AddParameters(task->name() + "_task_weighting", task->dim());

  // Set weighting to task
  program_.SetParameters(task->name() + "_task_weighting", task->Weighting());

  // Create objective for tracking
  casadi::SX xaccd_sym = casadi::SX::sym("xaccd", task->dim());
  casadi::SX w_sym = casadi::SX::sym("w", task->dim());

  // Create weighted objective as || xacc - xacc_d ||^2_W
  casadi::SX e = task->acc_sym() - xaccd_sym;
  sym::Expression obj = mtimes(mtimes(e.T(), casadi::SX::diag(w_sym)), e);

  // Add any additional parameters
  casadi::SXVector ps = {symbolic_terms_->qpos(), symbolic_terms_->qvel(),
                         xaccd_sym, w_sym};
  for (auto &pi : task->SymbolicParameters()) {
    ps.push_back(pi);
  }

  // Set inputs to the expression
  obj.SetInputs({symbolic_terms_->qacc()}, ps);

  // Add objective to program
  std::shared_ptr<opt::QuadraticCost> task_cost =
      std::make_shared<opt::QuadraticCost>(task->name() + "_motion_obj", obj);

  // Append any additional parameters
  sym::ParameterRefVector pv = {program_.GetParameters("qpos"),
                                program_.GetParameters("qvel"), xaccd, w};
  for (auto &pi : task->Parameters()) {
    pv.push_back(pi);
  }

  program_.AddQuadraticCost(task_cost, {variables_->qacc()}, pv);

  // Add motion task and corresponding acceleration reference parameter
  motion_tasks_.push_back(task);
  motion_tasks_weighting_.push_back(w);
  motion_tasks_acc_ref_.push_back(xaccd);
}

void OSC::AddContactPoint(const std::shared_ptr<ContactTask> &task) {
  /* Add constraint forces to program */
  sym::VariableVector lambda =
      sym::CreateVariableVector(task->name() + "_lambda", task->dim());
  // Add to variables
  variables_->AddConstraintForces(lambda);
  // Add to program
  program_.AddDecisionVariables(lambda);
  // Add bounds to constraint forces
  opt::Binding<opt::BoundingBoxConstraint> force_bounds =
      program_.AddBoundingBoxConstraint(task->fmin(), task->fmax(), lambda);

  /* Add constraints to the dynamics of the system */
  casadi::SX lam = casadi::SX::sym(task->name() + "_lambda", task->dim());
  symbolic_terms_->AddConstraintForces(lam);
  // Get constraint Jacobian
  casadi::SX J = jacobian(task->vel_sym(), symbolic_terms_->qvel());
  // Add joint-space forces based on the constraints
  AddConstraintsToDynamics(lam, lambda, J, task->SymbolicParameters(),
                           task->Parameters());

  /* Create parameters for task objective */
  Eigen::Ref<Eigen::MatrixXd> xaccd =
      program_.AddParameters(task->name() + "_contact_acc_ref", task->dim());
  // Add weighting for each task
  Eigen::Ref<Eigen::MatrixXd> w = program_.AddParameters(
      task->name() + "_contact_task_weighting", task->dim());
  // Set weighting to task
  program_.SetParameters(task->name() + "_contact_task_weighting",
                         task->Weighting());

  /* Create objective */
  // Create weighted objective as || xacc - xacc_d ||^2_W
  casadi::SX xaccd_sym = casadi::SX::sym("xaccd", task->dim());
  casadi::SX w_sym = casadi::SX::sym("w", task->dim());
  casadi::SX e = task->acc_sym() - xaccd_sym;
  sym::Expression obj = mtimes(mtimes(e.T(), casadi::SX::diag(w_sym)), e);

  // Set inputs to the expression
  obj.SetInputs(
      {symbolic_terms_->qacc()},
      {symbolic_terms_->qpos(), symbolic_terms_->qvel(), xaccd_sym, w_sym});

  // Add objective to program
  std::shared_ptr<opt::QuadraticCost> task_cost =
      std::make_shared<opt::QuadraticCost>(task->name() + "_contact", obj);

  program_.AddQuadraticCost(task_cost, {variables_->qacc()},
                            {program_.GetParameters("qpos"),
                             program_.GetParameters("qvel"), xaccd, w});

  // Create parameters for contact surface
  program_.AddParameters(task->name() + "_normal", 3);
  program_.AddParameters(task->name() + "_friction_mu", 1);

  // Set the parameters
  program_.SetParameters(task->name() + "_normal", task->normal());
  program_.SetParameters(task->name() + "_friction_mu",
                         Eigen::Vector<double, 1>(task->mu()));

  // Add friction constraint
  program_.AddLinearConstraint(
      friction_cone_con_, {lambda},
      {program_.GetParameters(task->name() + "_normal"),
       program_.GetParameters(task->name() + "_friction_mu")});

  // Add data to vectors for the OSC
  contact_tasks_.push_back(task);
  contact_tasks_acc_ref_.push_back(xaccd);
  contact_tasks_weighting_.push_back(w);
  contact_force_bounds_.push_back(force_bounds);
}

void OSC::AddHolonomicConstraint(const std::string &name, const casadi::SX &c,
                                 const casadi::SX &dcdt,
                                 const casadi::SX &d2cdt2) {
  // Create linear constraint
  casadi::SX A, b;
  casadi::SX::linear_coeff(d2cdt2, symbolic_terms_->qacc(), A, b, true);
  auto con = std::make_shared<opt::LinearConstraint>(
      name, A, b,
      casadi::SXVector({symbolic_terms_->qpos(), symbolic_terms_->qvel()}),
      opt::BoundsType::kEquality);

  // Add constraint to program
  program_.AddLinearConstraint(
      con, {variables_->qacc()},
      {program_.GetParameters("qpos"), program_.GetParameters("qvel")});

  // Add constraint forces as decision variables in the program
  sym::VariableVector lambda =
      sym::CreateVariableVector(name + "_lambda", c.size1());
  variables_->AddConstraintForces(lambda);
  program_.AddDecisionVariables(lambda);

  // Create symbolic representation of the constraint forces
  casadi::SX lam = casadi::SX::sym("lam", c.size1());
  symbolic_terms_->AddConstraintForces(lam);
  // Get constraint Jacobian
  casadi::SX J = jacobian(dcdt, symbolic_terms_->qvel());
  // Add joint-space forces to dynamics based on the constraints
  AddConstraintsToDynamics(lam, lambda, J, {}, {});
}

}  // namespace osc
}  // namespace control
}  // namespace damotion
