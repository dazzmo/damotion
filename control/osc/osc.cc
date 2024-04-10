#include "control/osc/osc.h"

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
}

void OSC::AddMotionTask(const std::shared_ptr<MotionTask> &task) {
    // Create objective for this task

    // Create parameter for program for task-acceleration error
    // minimisation
    Eigen::Ref<const Eigen::MatrixXd> xaccd =
        program_.AddParameters(task->name() + "_xaccd", task->dim());

    // Create objective for tracking
    casadi::SX xaccd_sym = casadi::SX::sym("xacc_d", task->dim());

    // Create objective as || xacc - xacc_d ||^2
    sym::Expression obj = casadi::SX::dot(task->acc_sym() - xaccd_sym,
                                          task->acc_sym() - xaccd_sym);

    // Add any additional parameters
    casadi::SXVector ps = {symbolic_terms_->qpos(), symbolic_terms_->qvel(),
                           xaccd_sym};
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
                                  program_.GetParameters("qvel"), xaccd};
    for (auto &pi : task->Parameters()) {
        pv.push_back(pi);
    }

    program_.AddQuadraticCost(task_cost, {variables_->qacc()}, pv);

    // Add motion task to the map
    motion_tasks_.push_back(task);
}

void OSC::AddContactPoint(std::shared_ptr<ContactTask> &task) {
    // Add new variables to the program
    sym::VariableVector lambda =
        sym::CreateVariableVector(task->name() + "_lambda", task->dim());
    // Add to variables
    variables_->AddConstraintForces(lambda);
    // Add to program
    program_.AddDecisionVariables(lambda);

    // Add bounds to constraint forces
    program_.AddBoundingBoxConstraint(task->fmin, task->fmax, lambda);

    // Add parameters for weighting and tracking?
    Eigen::Ref<const Eigen::MatrixXd> xaccd =
        program_.AddParameters(task->name() + "_xaccd", task->dim());

    // Create objective for tracking
    casadi::SX xaccd_sym = casadi::SX::sym("xacc_d", task->dim());

    // Create parameters for contact surface
    program_.AddParameters(task->name() + "_friction_mu", 1);
    program_.AddParameters(task->name() + "_normal", 3);

    sym::Expression obj;
    casadi::SX acc = task->Frame().acc_sym();
    // Create objective as || xacc - xacc_d ||^2
    obj = casadi::SX::dot(acc(casadi::Slice(0, 3)) - xaccd_sym,
                          acc(casadi::Slice(0, 3)) - xaccd_sym);

    // Set inputs to the expression
    obj.SetInputs(
        {symbolic_terms_->qacc()},
        {symbolic_terms_->qpos(), symbolic_terms_->qvel(), xaccd_sym});

    // Add objective to program
    std::shared_ptr<opt::QuadraticCost> task_cost =
        std::make_shared<opt::QuadraticCost>(task->name() + "_contact", obj);

    program_.AddQuadraticCost(task_cost, {variables_->qacc()},
                              {program_.GetParameters("qpos"),
                               program_.GetParameters("qvel"), xaccd});

    // Create projected dynamics

    // Create symbolic representation of the constraint forces
    casadi::SX lam = casadi::SX::sym("lam", task->dim());
    symbolic_terms_->AddConstraintForces(lam);
    // Get constraint Jacobian
    casadi::SX J = jacobian(task->Frame().vel_sym(), symbolic_terms_->qvel());
    // Add joint-space forces based on the constraints
    AddConstraintsToDynamics(lam, lambda, J, task->SymbolicParameters(),
                             task->Parameters());

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