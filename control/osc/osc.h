#ifndef OSC_OSC_H
#define OSC_OSC_H

#include <Eigen/Core>
#include <algorithm>
#include <casadi/casadi.hpp>
#include <map>
#include <string>

#include "common/profiler.h"
#include "solvers/constraint.h"
#include "solvers/cost.h"
#include "solvers/program.h"
#include "system/constraint.h"
#include "utils/casadi.h"
#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"
#include "utils/pinocchio_model.h"

namespace opt = damotion::optimisation;
namespace sym = damotion::symbolic;
namespace utils = damotion::utils;

namespace damotion {
namespace control {
namespace osc {

// TODO - Split these into different files (e.g. tasks, costs, OSC etc...)

typedef utils::casadi::PinocchioModelWrapper::TargetFrame TargetFrame;

class Task {
   public:
    Task() = default;
    ~Task() = default;

    /**
     * @brief Dimension of the task.
     *
     * @return const int
     */
    const int dim() const { return dim_; }

    /**
     * @brief Resizes the dimension of the task.
     *
     * @param ndim
     */
    void Resize(const int ndim) {
        dim_ = ndim;
        e_ = Eigen::VectorXd::Zero(ndim);
        de_ = Eigen::VectorXd::Zero(ndim);
        Kp_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(ndim);
        Kp_.setZero();
        Kd_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(ndim);
        Kd_.setZero();
    }

    // void SetReference(const Eigen::VectorXd &xr);
    // void SetReference(const Eigen::Quaterniond &qr);
    // void SetReference(const Eigen::VectorXd &xr, const Eigen::Quaterniond
    // &qr);

    const Eigen::VectorXd &Error() { return e_; }
    const Eigen::VectorXd &ErrorDerivative() { return de_; }

    void SetKpGains(const Eigen::VectorXd &Kp) { Kp_.diagonal() = Kp; }
    void SetKdGains(const Eigen::VectorXd &Kd) { Kd_.diagonal() = Kd; }

    virtual Eigen::VectorXd GetPDError() { return Kp_ * e_ + Kd_ * de_; }

   protected:
    // Task error
    Eigen::VectorXd e_;
    Eigen::VectorXd de_;
    // PD Tracking gains
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kp_;
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kd_;

    // Desired pose translational component
    Eigen::Vector3d xr_;
    // Desired pose rotational component
    Eigen::Quaterniond qr_;

    // Desired pose velocity translational component
    Eigen::Vector3d vr_ = Eigen::Vector3d::Zero();
    // Desired pose velocity rotational component
    Eigen::Vector3d wr_ = Eigen::Vector3d::Zero();

   private:
    int dim_ = 0;
};

/**
 * @brief Motion task for operational space control, typically used for tracking
 * of a frame
 *
 */
class MotionTask : public Task {
   public:
    enum class Type { kTranslational, kRotational, kFull };

    MotionTask() = default;
    ~MotionTask() = default;

    MotionTask(const std::shared_ptr<TargetFrame> &frame, const Type &type)
        : Task() {
        type_ = type;
        if (type == Type::kTranslational || type == Type::kRotational) {
            Resize(3);
        } else if (type == Type::kFull) {
            Resize(6);
        }
    }

    /**
     * @brief Returns the type of the motion task, which can be translational,
     * rotational or full
     *
     * @return const Type&
     */
    const Type &type() const { return type_; }

    /**
     * @brief The TargetFrame the motion task is designed for
     *
     * @return const TargetFrame&
     */
    TargetFrame &Frame() { return *frame_; }

    /**
     * @brief Compute the desired tracking task acceleration as a PD error
     * metric on the task position and velocity errors
     *
     * @return Eigen::VectorXd
     */
    void ComputePoseError();

   private:
    Type type_;
    // Target frame the motion task is associated with
    std::shared_ptr<TargetFrame> frame_;
};

/**
 * @brief A contact task that represents a rigid contact for a given frame
 * within the environment or an object
 *
 */
class ContactTask : public Task {
   public:
    ContactTask() = default;
    ~ContactTask() = default;

    ContactTask(const std::shared_ptr<TargetFrame> &frame) : Task() {
        Resize(3);
    }

    /**
     * @brief The TargetFrame the motion task is designed for
     *
     * @return const TargetFrame&
     */
    TargetFrame &Frame() { return *frame_; }

    // Desired pose translational component
    Eigen::Vector3d xr;

    // Whether the point is in contact or not
    bool inContact = false;

    // Contact surface normal
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();

    // Friction coefficient
    double mu = 1.0;

    // Maximum allowable force
    Eigen::Vector3d fmax = Eigen::Vector3d::Ones();
    // Minimum allowable force
    Eigen::Vector3d fmin = Eigen::Vector3d::Ones();

   private:
    std::shared_ptr<TargetFrame> frame_;
};

optimisation::Constraint LinearisedFrictionConstraint() {
    // Square pyramid approximation
    casadi::SX lambda = casadi::SX::sym("lambda", 3);
    casadi::SX mu = casadi::SX::sym("mu");

    casadi::SX l_x = lambda(0), l_y = lambda(1), l_z = lambda(2);

    // TODO - Add parameter for surface normal

    // Friction cone constraint with square pyramid approximation
    sym::Expression cone;
    cone = casadi::SX(4, 1);
    cone(0) = sqrt(2.0) * l_x + mu * l_z;
    cone(1) = -sqrt(2.0) * l_x - mu * l_z;
    cone(2) = sqrt(2.0) * l_y + mu * l_z;
    cone(3) = -sqrt(2.0) * l_y - mu * l_z;

    cone.SetInputs({lambda}, {mu});
    return optimisation::Constraint("friction_cone", cone,
                                    opt::BoundsType::kPositive);
}

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
     * @brief Adds a generic contact task that relies on the conventional state
     * of the system (qpos, qvel, qacc) and no other parameters. If more
     * parameters are required for the task, we suggest adding these parameters,
     * constraints and objectives manually by accessing the program through
     * GetProgram().
     *
     * @param name
     * @param task
     */
    void AddContactPoint(const std::string &name,
                         std::shared_ptr<ContactTask> &task);

    /**
     * @brief Adds a generic motion task that relies on the conventional state
     * of the system (qpos, qvel, qacc) and no other parameters. If more
     * parameters are required for the task, we suggest adding these parameters,
     * constraints and objectives manually by accessing the program through
     * GetProgram().
     *
     * @param name
     * @param task
     * @return void
     */
    void AddMotionTask(const std::string &name,
                       std::shared_ptr<MotionTask> &task);

    /**
     * @brief For a constraint that can be written in the form h(q) = 0,
     * computes the linear constraint imposed by the second derivative in time
     * of the constraint. This function assumes the input variables to the
     * constraint are qpos, qvel and qacc of the program with no other
     * parameters. If a constraint with extra parameters is required, consider
     * adding it manually to the program by GetProgram().
     *
     * @param name
     * @param c The holonomic constraint c(q) = 0
     * @param dcdt The first derivative of c(q) wrt time
     * @param d2cdt2 The second derivative of c(q) wrt time
     */
    void AddHolonomicConstraint(const std::string &name, const casadi::SX &c,
                                const casadi::SX &dcdt,
                                const casadi::SX &d2cdt2);

    /**
     * @brief Sets the unconstrained inverse dynamics for the OSC program. Note
     * that this must be called before adding any other tasks so that constraint
     * forces can be added to this expression.
     *
     * @param dynamics
     */
    void AddUnconstrainedInverseDynamics(const casadi::SX &dynamics) {
        constrained_dynamics_ = dynamics;
        // Initialise inputs to the system
        constrained_dynamics_sym_ = casadi::SX::vertcat(
            {symbolic_terms_->qacc(), symbolic_terms_->ctrl()});
        constrained_dynamics_var_.resize(variables_->qacc().size() +
                                         variables_->ctrl().size());
        constrained_dynamics_var_ << variables_->qacc(), variables_->ctrl();
    }

    /**
     * @brief Creates the program after adding all necessary tasks, constraints
     * and objectives
     *
     */
    void CreateProgram() {
        // Create linear constraint for the constrained dynamics
        casadi::SX A, b;
        casadi::SX::linear_coeff(constrained_dynamics_,
                                 constrained_dynamics_sym_, A, b, true);
        auto con = std::make_shared<opt::LinearConstraint>(
            "dynamics", A, b,
            casadi::SXVector(
                {symbolic_terms_->qpos(), symbolic_terms_->qvel()}),
            opt::BoundsType::kEquality);
        // Add constraint to program
        program_.AddLinearConstraint(
            con, {variables_->qacc()},
            {program_.GetParameters("qpos"), program_.GetParameters("qvel")});

        // Construct the decision variable vector  [qacc, ctrl, lambda]
        sym::VariableVector x(program_.NumberOfDecisionVariables());
        // Set it as qacc, ctrl, constraint forces
        x.topRows(variables_->qacc().size() + variables_->ctrl().size())
            << variables_->qacc(),
            variables_->ctrl();
        // Add any constraint forces
        int idx = variables_->qacc().size() + variables_->ctrl().size();
        for (int i = 0; i < variables_->lambda().size(); ++i) {
            x.middleRows(idx, variables_->lambda(i).size()) =
                variables_->lambda(i);
        }
        program_.SetDecisionVariableVector(x);
    }

    void Update(const Eigen::VectorXd &qpos, const Eigen::VectorXd &qvel) {
        program_.SetParameters("qpos", qpos);
        program_.SetParameters("qvel", qvel);

        // Update frame positions using the given data

        // Update each task given the updated data and references
        for (auto &task : motion_tasks_) {
            // Set desired task accelerations
            task.second->ComputePoseError();
            program_.SetParameters(task.first + "_xaccd",
                                   task.second->GetPDError());
        }

        for (auto &task : contact_tasks_) {
            // Check if the frame is in contact
            if (task.second->inContact) {
                // Update forces for the task
            } else {
                // Update forces for the task
            }
        }

        // Program is updated
    }

    /**
     * @brief Get the underlying program for the OSC. Allows you to add
     * additional variables, parameters, costs, constraints that are not
     * included in conventional OSC programs.
     *
     */
    opt::Program &GetProgram() { return program_; }

    /**
     * @brief Updates the contact data within the program for the contact task
     * task
     *
     * @param name
     * @param task
     */
    void UpdateContactPoint(const std::string &name, ContactTask &task) {
        // Update contact normal, friction cone etc...
        program_.SetParameters(name + "_normal", task.normal);
    }

    class Variables {
       public:
        Variables() = default;
        ~Variables() = default;

        Variables(int nq, int nv, int nu) {
            qacc_ = sym::CreateVariableVector("qacc", nv),
            ctrl_ = sym::CreateVariableVector("ctrl", nu);
            lambda_ = {};
        }

        const sym::VariableVector &qacc() const { return qacc_; }
         sym::VariableVector &qacc()  { return qacc_; }
         
        const sym::VariableVector &ctrl() const { return ctrl_; }
        sym::VariableVector &ctrl() { return ctrl_; }

        const std::vector<sym::VariableVector> &lambda() const {
            return lambda_;
        }

        const sym::VariableVector &lambda(const int i) const {
            return lambda_[i];
        }

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
            qpos_ = casadi::SX::sym("qpos", nq),
            qvel_ = casadi::SX::sym("qvel", nv),
            qacc_ = casadi::SX::sym("qacc", nv),
            ctrl_ = casadi::SX::sym("ctrl", nu);
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
    // TODO - Add additional parameters to the constraint where necessary
    void AddConstraintsToDynamics(const casadi::SX &lambda,
                                  const sym::VariableVector &lambda_var,
                                  const casadi::SX &J) {
        // Add variables to the input vectors
        constrained_dynamics_sym_ =
            casadi::SX::vertcat({constrained_dynamics_sym_, lambda});
        constrained_dynamics_var_.conservativeResize(
            constrained_dynamics_var_.size() + lambda_var.size());
        constrained_dynamics_var_.bottomRows(lambda_var.size()) << lambda_var;
        // Add to the constraint
        constrained_dynamics_ -= mtimes(J.T(), lambda);
    }

    // Conventional optimisation variables
    std::unique_ptr<Variables> variables_;

    // Symbolic variables used for
    std::unique_ptr<SymbolicTerms> symbolic_terms_;

    std::unordered_map<std::string, std::shared_ptr<MotionTask>> motion_tasks_;
    std::unordered_map<std::string, std::shared_ptr<ContactTask>>
        contact_tasks_;

    // Constrained dynamics
    casadi::SX constrained_dynamics_sym_;
    casadi::SX constrained_dynamics_par_sym_;
    sym::VariableVector constrained_dynamics_var_;
    sym::Expression constrained_dynamics_;

    // Underlying program for the OSC
    opt::Program program_;
};

}  // namespace osc
}  // namespace control
}  // namespace damotion

#endif/* OSC_OSC_H */
