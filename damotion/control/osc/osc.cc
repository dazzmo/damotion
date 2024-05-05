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

std::shared_ptr<opt::LinearConstraint<Eigen::MatrixXd>>
LinearisedFrictionConstraint() {
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

  return std::make_shared<opt::LinearConstraint<Eigen::MatrixXd>>(
      "friction_cone", cone, opt::BoundsType::kPositive);
}

OSC::OSC(int nq, int nv, int nu) : nq_(nq), nv_(nv), nu_(nu) {
  // Create mathematical program
  program_ = opt::Program("osc");
  // Create friction cone constraint to be bounded to by contact tasks
  friction_cone_con_ = LinearisedFrictionConstraint();
}

void OSC::AddMotionTask(std::shared_ptr<MotionTask> &task,
                        model::symbolic::TargetFrame &frame) {
  // Register frame with task
  task->SetFrame(frame.CreateFrame());
  /* Motion task parameters */

  sym::ParameterVector xaccd = sym::CreateParameterVector(
                           task->name() + "_motion_acc_ref", task->dim()),
                       w = sym::CreateParameterVector(
                           task->name() + "_motion_task_weighting",
                           task->dim());

  program_.AddParameters(xaccd);
  program_.AddParameters(w);

  // Set references and values
  MotionTaskParameters parameters(program_.GetParameterRef(xaccd),
                                  program_.GetParameterRef(w));

  parameters.w << task->Weighting();

  // Create objective for tracking
  casadi::SX xaccd_sx = casadi::SX::sym("xaccd", task->dim());
  casadi::SX w_sx = casadi::SX::sym("w", task->dim());

  // Create weighted objective as || xacc - xacc_d ||^2_W
  casadi::SX e = frame.acc() - xaccd_sx;
  sym::Expression obj = mtimes(mtimes(e.T(), casadi::SX::diag(w_sx)), e);

  // Set inputs to the expression
  obj.SetInputs({qacc_sx_}, {qpos_sx_, qvel_sx_, xaccd_sx, w_sx});

  // Add objective to program
  opt::QuadraticCost<Eigen::MatrixXd>::SharedPtr task_cost =
      std::make_shared<opt::QuadraticCost<Eigen::MatrixXd>>(
          task->name() + "_motion_obj", obj);

  program_.AddQuadraticCost(task_cost, {qacc_}, {qpos_, qvel_, xaccd, w});

  // Add motion task and corresponding acceleration reference parameter
  motion_tasks_.push_back(task);
  motion_task_parameters_.push_back(parameters);
}

void OSC::AddContactTask(std::shared_ptr<ContactTask> &task,
                         model::symbolic::TargetFrame &frame) {
  // Register frame with task
  task->SetFrame(frame.CreateFrame());

  /* Contact task parameters */
  sym::ParameterVector xacc = sym::CreateParameterVector(
                           task->name() + "_contact_acc_ref", task->dim()),
                       w = sym::CreateParameterVector(
                           task->name() + "_contact_task_weighting",
                           task->dim()),
                       normal = sym::CreateParameterVector(
                           task->name() + "_normal", 3),
                       mu = sym::CreateParameterVector(
                           task->name() + "_friction_mu", 1);

  program_.AddParameters(xacc);
  program_.AddParameters(w);
  program_.AddParameters(normal);
  program_.AddParameters(mu);

  // Set parameters
  ContactTaskParameters parameters(
      program_.GetParameterRef(xacc), program_.GetParameterRef(w),
      program_.GetParameterRef(mu), program_.GetParameterRef(normal));

  parameters.w << task->Weighting();
  parameters.normal << task->normal();
  parameters.mu << task->mu();

  /* Add constraint forces to program */
  sym::VariableVector lambda =
      sym::CreateVariableVector(task->name() + "_lambda", task->dim());
  casadi::SX lambda_sx = casadi::SX::sym(task->name() + "_lambda", task->dim());
  AddConstraintForceVariables(lambda, lambda_sx);

  // Add bounds to constraint forces
  opt::Binding<opt::BoundingBoxConstraint<Eigen::MatrixXd>> force_bounds =
      program_.AddBoundingBoxConstraint(task->fmin(), task->fmax(), lambda);

  // Get constraint Jacobian
  casadi::SX J = jacobian(frame.vel(), qvel_sx_);
  // Add joint-space forces based on the constraints
  constrained_dynamics_ -= mtimes(J.T(), lambda_sx);

  /* Create objective */

  // Create weighted objective as || xacc - xacc_d ||^2_W
  casadi::SX xaccd_sx = casadi::SX::sym("xaccd", task->dim());
  casadi::SX w_sx = casadi::SX::sym("w", task->dim());
  casadi::SX e = frame.acc() - xaccd_sx;
  sym::Expression obj = mtimes(mtimes(e.T(), casadi::SX::diag(w_sx)), e);

  // Set inputs to the expression
  obj.SetInputs({qacc_sx_}, {qpos_sx_, qvel_sx_, xaccd_sx, w_sx});

  // Add objective to program
  std::shared_ptr<opt::QuadraticCost<Eigen::MatrixXd>> task_cost =
      std::make_shared<opt::QuadraticCost<Eigen::MatrixXd>>(
          task->name() + "_contact", obj);

  program_.AddQuadraticCost(task_cost, {qacc_}, {qpos_, qvel_, xacc, w});

  // Add friction constraint
  program_.AddLinearConstraint(friction_cone_con_, {lambda}, {normal, mu});

  // Add data to vectors for the OSC
  contact_tasks_.push_back(task);
  contact_task_parameters_.push_back(parameters);
  contact_force_bounds_.push_back(force_bounds);
}

void OSC::AddHolonomicConstraint(const std::string &name, const casadi::SX &c,
                                 const casadi::SX &dcdt,
                                 const casadi::SX &d2cdt2) {
  // Create linear constraint
  casadi::SX A, b;
  casadi::SX::linear_coeff(d2cdt2, qacc_sx_, A, b, true);
  auto con = std::make_shared<opt::LinearConstraint<Eigen::MatrixXd>>(
      name, A, b, casadi::SXVector({qpos_sx_, qvel_sx_}),
      opt::BoundsType::kEquality);

  // Add constraint to program
  program_.AddLinearConstraint(con, {qacc_}, {qpos_, qvel_});

  // Add constraint forces as decision variables in the program
  sym::VariableVector lambda =
      sym::CreateVariableVector(name + "_lambda", c.size1());
  casadi::SX lambda_sx = casadi::SX::sym("lam", c.size1());
  AddConstraintForceVariables(lambda, lambda_sx);
  // Get constraint Jacobian
  casadi::SX J = jacobian(dcdt, qvel_sx_);
  // Add joint-space forces to dynamics based on the constraints
  constrained_dynamics_ -= mtimes(J.T(), lambda_sx);
}

}  // namespace osc
}  // namespace control
}  // namespace damotion
