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

void MotionTask::ComputePoseError() {
    damotion::common::Profiler profiler("MotionTask::ComputePoseError");

    const Eigen::VectorXd &xpos = Frame().pos(), &xvel = Frame().vel();

    if (type_ == Type::kTranslational) {
        e_ = xpos.topRows(3) - xr_;
        de_ = xvel.topRows(3) - vr_;
    } else if (type_ == Type::kRotational) {
        // Get rotational component
        Eigen::Vector4d qv = xpos.bottomRows(4);
        Eigen::Quaterniond q(qv);
        // Compute rotational error
        Eigen::Matrix3d R =
            qr_.toRotationMatrix().transpose() * q.toRotationMatrix();
        double theta = 0.0;
        Eigen::Vector3d elog;
        elog = pinocchio::log3(R, theta);
        // Compute rate of error
        Eigen::Matrix3d Jlog;
        pinocchio::Jlog3(theta, elog, Jlog);
        e_ = elog;
        de_ = Jlog * xvel - wr_;
    } else {
        // Convert to 6D pose
        Eigen::Vector3d x = xpos.topRows(3);
        Eigen::Vector4d qv = xpos.bottomRows(4);
        Eigen::Quaterniond q(qv);
        pinocchio::SE3 se3r(qr_.toRotationMatrix(), xr_),
            se3(q.toRotationMatrix(), x);

        e_ = pinocchio::log6(se3r.actInv(se3)).toVector();
        Eigen::Matrix<double, 6, 6> Jlog;
        pinocchio::Jlog6(se3r.actInv(se3), Jlog);
        de_ = Jlog * xvel;
        de_.topRows(3) -= vr_;     // Translational component
        de_.bottomRows(3) -= wr_;  // Rotational component
    }
}

OSC::OSC(int nq, int nv, int nu) {
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

void OSC::AddMotionTask(const std::string &name,
                        std::shared_ptr<MotionTask> &task) {
    // Create objective for this task

    // Create parameter for program for task-acceleration error
    // minimisation
    Eigen::Ref<const Eigen::MatrixXd> xaccd =
        program_.AddParameters(name + "_xaccd", task->dim());

    // Create objective for tracking
    casadi::SX xaccd_sym = casadi::SX::sym("xacc_d", task->dim());

    sym::Expression obj;
    casadi::SX acc, acc_full = task->Frame().acc_sym();

    if (task->type() == MotionTask::Type::kTranslational) {
        acc = acc_full(casadi::Slice(0, 3));
    } else if (task->type() == MotionTask::Type::kRotational) {
        acc = acc_full(casadi::Slice(3, 6));
    } else {
        acc = acc_full;
    }
    // Create objective as || xacc - xacc_d ||^2
    obj = casadi::SX::dot(acc - xaccd_sym, acc - xaccd_sym);
    // Set inputs to the expression
    obj.SetInputs(
        {symbolic_terms_->qacc()},
        {symbolic_terms_->qpos(), symbolic_terms_->qvel(), xaccd_sym});

    // Add objective to program
    std::shared_ptr<opt::QuadraticCost> task_cost =
        std::make_shared<opt::QuadraticCost>(name + "_tracking", obj);

    program_.AddQuadraticCost(task_cost, {variables_->qacc()},
                              {program_.GetParameters("qpos"),
                               program_.GetParameters("qvel"), xaccd});

    // Add motion task to the map
    motion_tasks_[name] = task;
}

void OSC::AddContactPoint(const std::string &name,
                          std::shared_ptr<ContactTask> &task) {
    // Add new variables to the program
    sym::VariableVector lambda =
        sym::CreateVariableVector(name + "_lambda", task->dim());
    variables_->AddConstraintForces(lambda);
    program_.AddDecisionVariables(lambda);
    // Add bounds to constraint forces
    program_.AddBoundingBoxConstraint(task->fmin, task->fmax, lambda);

    // Add parameters for weighting and tracking?
    Eigen::Ref<const Eigen::MatrixXd> xaccd =
        program_.AddParameters(name + "_xaccd", task->dim());

    // Create objective for tracking
    casadi::SX xaccd_sym = casadi::SX::sym("xacc_d", task->dim());

    // Create parameters for contact surface
    // TODO - Update these via a function
    program_.AddParameters(name + "_friction_mu", 1);
    program_.AddParameters(name + "_normal", 3);

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
        std::make_shared<opt::QuadraticCost>(name + "_contact", obj);

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
    AddConstraintsToDynamics(lam, lambda, J);

    // TODO - Add to the input variables for the dynamics
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
    AddConstraintsToDynamics(lam, lambda, J);
}

}  // namespace osc
}  // namespace control
}  // namespace damotion