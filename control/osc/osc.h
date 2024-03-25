#ifndef OSC_OSC_H
#define OSC_OSC_H

#include <Eigen/Core>
#include <algorithm>
#include <casadi/casadi.hpp>
#include <map>
#include <string>

#include "solvers/program.h"
#include "utils/casadi.h"
#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"

// ? create functions that compute the necessary constraints and stuff for a
// ? problem, then pass it to a given solver?

#include "common/profiler.h"
#include "system/constraint.h"

namespace damotion {
namespace control {

class OSCController : public optimisation::Program {
   public:
    OSCController() = default;
    ~OSCController() = default;

    OSCController(int nq, int nv, int nu) : nq_(nq), nv_(nv) {
        // Add default variables to the problem
        AddDecisionVariables("qacc", nv);
        AddDecisionVariables("ctrl", nu);
        AddDecisionVariables("lam", 0);

        // Default parameters to the problem
        AddParameters("qpos", nq);
        AddParameters("qvel", nv);
    }

    // Holonomic Constraint
    class HolonomicConstraint {
       public:
        HolonomicConstraint() = default;
        ~HolonomicConstraint() = default;

        void SetName(const std::string &name) { name_ = name; }
        const std::string &name() const { return name_; }

        void SetDimension(const int &dim) { dim_ = dim; }
        const int &Dimension() const { return dim_; }

        void SetConstraintForceIndex(const int &idx) { lam_idx_ = idx; }
        const int ConstraintForceIndex() const { return lam_idx_; }

        void SetFunction(const casadi::Function &f) {
            f_ = utils::casadi::FunctionWrapper(f);
        }

        /**
         * @brief Function that computes the holonomic constraint c(q) and its
         * first and second derivatives in time.
         *
         * @return utils::casadi::FunctionWrapper&
         */
        utils::casadi::FunctionWrapper &Function() { return f_; }

       private:
        int dim_ = 0;
        int lam_idx_ = -1;
        std::string name_;
        utils::casadi::FunctionWrapper f_;
    };

    // Associated with tracking a given point in SE(3)
    class TrackingTask : public HolonomicConstraint {
       public:
        enum class Type { kTranslational, kRotational, kFull };

        TrackingTask() = default;
        ~TrackingTask() = default;

        Type type;
        // Tracking gains
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kp;
        // Tracking gains
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kd;

        // Error in pose
        Eigen::VectorXd e;
        Eigen::VectorXd de;

        // Desired pose translational component
        Eigen::Vector3d xr;
        // Desired pose rotational component
        Eigen::Quaterniond qr;

        // Desired pose velocity translational component
        Eigen::Vector3d vr = Eigen::Vector3d::Zero();
        // Desired pose velocity rotational component
        Eigen::Vector3d wr = Eigen::Vector3d::Zero();

        Eigen::VectorXd ComputeDesiredAcceleration();
    };

    // Associated with point-contact with a given surface
    class ContactTask : public HolonomicConstraint {
       public:
        ContactTask() = default;
        ~ContactTask() = default;

        // Whether the point is in contact or not
        bool inContact = false;
        // Contact surface normal
        Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();

        // Friction coefficient
        double mu = 1.0;

        // Point to keep in contact with
        Eigen::Vector3d xr;
    };

    /**
     * @brief Creates a holonomic constraint based on the function c, adding any
     * additional parameters to the program.
     *
     * @param name
     * @param c
     * @param constraint
     */
    void CreateHolonomicConstraint(const std::string &name, casadi::Function &c,
                                   HolonomicConstraint &constraint) {
        constraint.SetName(name);

        // Add any parameters to the parameter map
        for (int i = 0; i < c.n_in(); ++i) {
            if (!IsDecisionVariable(c.name_in(i))) {
                AddParameters(c.name_in(i), c.size1_in(i));
            }
        }

        // Wrap function
        constraint.SetFunction(c);
    }

    void AddTrackingTask(const std::string &name, casadi::Function &x,
                         const TrackingTask::Type &type) {
        // Create new task
        TrackingTask task;
        CreateHolonomicConstraint(name, x, task);

        if (task.type == TrackingTask::Type::kRotational ||
            task.type == TrackingTask::Type::kTranslational) {
            task.SetDimension(3);
        } else {
            task.SetDimension(6);
        }
        // Add tracking task to map
        tracking_tasks_[name] = task;
    }

    void AddContactTask(const std::string &name, casadi::Function &x) {
        // Create new task
        ContactTask task;
        CreateHolonomicConstraint(name, x, task);

        // Set the dimension of the task
        task.SetDimension(3);

        // Add tracking task to map
        contact_tasks_[name] = task;
    }

    /**
     * @brief Adds a holonomic constraint \f$ c(q) = 0 \f$ to the OSC program.
     * Function input c requires the constraint and its first two derivatives in
     * time (i.e. \f$ c(q) , \dot{c}(q, \dot{q}), \ddot{c}(q, \dot{q},
     * \ddot{q})\f$)
     *
     * @param name
     * @param c Input function that computes the constraint and time derivatives
     */
    void AddHolonomicConstraint(const std::string &name, casadi::Function &c) {
        // Create new constraint
        HolonomicConstraint constraint;
        CreateHolonomicConstraint(name, c, constraint);
        // Set the dimension of the task
        constraint.SetDimension(c.size1_out(0));
        // Add holonomic constraint to map
        holonomic_constraints_[name] = constraint;
    }

    void AddSystemDynamicsFunction(casadi::Function &f) { fdyn_ = f; }

    TrackingTask &GetTrackingTask(const std::string &name) {
        if (tracking_tasks_.find(name) == tracking_tasks_.end()) {
            throw std::runtime_error(name + "is not a tracking task");
        } else {
            return tracking_tasks_[name];
        }
    }

    ContactTask &GetContactTask(const std::string &name) {
        if (contact_tasks_.find(name) == contact_tasks_.end()) {
            throw std::runtime_error(name + "is not a tracking task");
        } else {
            return contact_tasks_[name];
        }
    }

    void SetUpTrackingTask(TrackingTask &task) {
        damotion::common::Profiler("OSCController::SetUpTrackingTask");


        // for(int i = 0; i)
        // Get necessary inputs
        casadi::StringVector inames =
            utils::casadi::CreateInputNames(task.Function().f());
        casadi::SXVector in = GetSymbolicFunctionInput(inames);

        // ! Only extract parameters? Make input {x, p}

        // Get second time derivative of constraint
        casadi::SX d2cdt2 = task.Function().f()(in)[2];

        // Add tracking acceleration as a parameter to the program
        AddParameters(task.name() + "_xacc_d", task.Dimension());

        // Create tracking error against desired task acceleration
        Eigen::VectorX<casadi::SX> xacc_e, xacc_d_e;
        utils::casadi::toEigen(d2cdt2, xacc_e);
        utils::casadi::toEigen(GetParameters(task.name() + "_xacc_d").sym(),
                               xacc_d_e);

        // Create tracking cost
        casadi::SX obj = 0;
        // Get necessary components of task acceleration
        if (task.type == TrackingTask::Type::kTranslational) {
            obj = (xacc_e.topRows(3) - xacc_d_e).squaredNorm();
        } else if (task.type == TrackingTask::Type::kRotational) {
            obj = (xacc_e.bottomRows(3) - xacc_d_e).squaredNorm();
        } else {
            obj = (xacc_e - xacc_d_e).squaredNorm();
        }

        in.push_back(GetParameters(task.name() + "_xacc_d").sym());

        // Add cost to the program
        AddCost(task.name() + "_tracking_cost", obj, in);

        // SetUp function data to task
        SetFunctionInputData(task.Function());
    }

    void SetUpContactTask(ContactTask &task) {
        damotion::common::Profiler("OSCController::SetUpContactTask");

        // Get necessary inputs
        casadi::StringVector inames =
            utils::casadi::CreateInputNames(task.Function().f());
        casadi::SXVector in = GetSymbolicFunctionInput(inames);

        // Get second time derivative of constraint
        casadi::SX d2cdt2 = task.Function().f()(in)[2];

        // Get translational component
        casadi::SX xacc = d2cdt2(casadi::Slice(0, 3));

        /* No-slip constraint */
        AddConstraint(task.name() + "_no_slip", xacc, in,
                      optimisation::BoundsType::kEquality);

        /* Friction cone constraint */
        // Get contact forces from force vector
        int idx = task.ConstraintForceIndex();
        casadi::SX lam =
            GetDecisionVariables("lam").sym()(casadi::Slice(idx, idx + 3));

        AddFrictionConeConstraint(task.name(), lam);

        // SetUp function data to task
        SetFunctionInputData(task.Function());
    }

    void SetUpHolonomicConstraint(HolonomicConstraint &con) {
        damotion::common::Profiler("OSCController::SetUpHolonomicConstraint");

        casadi::StringVector inames =
            utils::casadi::CreateInputNames(con.Function().f());
        casadi::SXVector in = GetSymbolicFunctionInput(inames);

        // Get second time derivative of constraint
        casadi::SX d2cdt2 = con.Function().f()(in)[2];
        // Set second rate of change of constraint to zero to ensure no change
        // Add constraint to program
        AddConstraint(con.name(), d2cdt2, in,
                      optimisation::BoundsType::kEquality);
    }

    // Given all information for the problem, initialise the program
    void Initialise() {
        // Determine how many contact wrenches need to be considered
        int nc = 0;
        for (auto &task : contact_tasks_) {
            // Assign index to task
            task.second.SetConstraintForceIndex(nc);
            nc += task.second.Dimension();
        }
        // Add any holonomic constraints
        // TODO - Make projection a possibility
        for (auto &con : holonomic_constraints_) {
            // Assign index
            con.second.SetConstraintForceIndex(nc);
            nc += con.second.Dimension();
        }

        // Resize lambda to account for all constraint forces
        ResizeDecisionVariables("lam", nc);

        // Create optimisation vector with given ordering
        ConstructDecisionVariableVector({"qacc", "ctrl", "lam"});

        // Set up contact constraints to program
        for (auto &p : contact_tasks_) {
            ContactTask &task = p.second;
            SetUpContactTask(task);
        }

        // SetUp tracking tasks to program
        for (auto &task : tracking_tasks_) {
            SetUpTrackingTask(task.second);
        }

        // SetUp holonomic constraints to program
        for (auto &con : holonomic_constraints_) {
            SetUpHolonomicConstraint(con.second);
        }

        // Compute constrained dynamics constraint and add to program
        ComputeConstrainedDynamics();

        // Construct the final constraint vector
        ConstructConstraintVector();

        // Create bounds for decision variables and constraints
        UpdateDecisionVariableVectorBounds();
        UpdateConstraintVectorBounds();
    }

    /**
     * @brief Create a linearised friction cone with tuneable parameter mu to
     * adjust the friction cone constraints
     *
     * @param lambda
     */
    void AddFrictionConeConstraint(const std::string &name,
                                   casadi::SX &lambda) {
        // Square pyramid approximation
        casadi::SX l_x = lambda(0), l_y = lambda(1), l_z = lambda(2);

        AddParameters(name + "_friction_mu", 1);
        casadi::SX mu = GetParameters(name + "_friction_mu").sym();

        // Friction cone constraint with square pyramid approximation
        casadi::SX cone(4, 1);
        cone(0) = sqrt(2.0) * l_x + mu * l_z;
        cone(1) = -sqrt(2.0) * l_x - mu * l_z;
        cone(2) = sqrt(2.0) * l_y + mu * l_z;
        cone(3) = -sqrt(2.0) * l_y - mu * l_z;

        casadi::SXVector in = {GetDecisionVariables("qacc").sym(),
                               GetDecisionVariables("ctrl").sym(),
                               GetDecisionVariables("lam").sym(),
                               GetParameters(name + "_friction_mu").sym()};

        AddConstraint(name + "_friction", cone, in,
                      optimisation::BoundsType::kPositive);

        // ! Need to add constraint bounds
    }

    void UpdateContactFrictionCoefficient(const std::string &name,
                                          const double &mu) {
        // ! Throw error if name is not in lookup
        Eigen::VectorXd val(1);
        val[0] = mu;
        SetParameters(name + "_friction_mu", val);
    }

    /**
     * @brief Adds any unique inputs from a new function into the inputs names
     * and in
     *
     * @param names
     * @param in
     * @param names_new
     * @param in_new
     */
    void AppendFunctionInputs(std::vector<std::string> &names,
                              casadi::SXVector &in,
                              std::vector<std::string> &names_new,
                              casadi::SXVector &in_new) {
        for (int i = 0; i < in_new.size(); ++i) {
            // Add any parameters that aren't in the function
            if (std::find(names.begin(), names.end(), names_new[i]) ==
                names.end()) {
                // Add variable/parameter to the function input
                names.push_back(names_new[i]);
                in.push_back(in_new[i]);
            }
        }
    }

    void ComputeConstrainedDynamics() {
        // Get inputs for function

        casadi::StringVector inames = utils::casadi::CreateInputNames(fdyn_);
        casadi::SXVector in = GetSymbolicFunctionInput(inames);

        // Evaluate dynamics with optimisation variables/parameters
        casadi::SX dyn = fdyn_(in)[0];

        // Any contact-tasks
        for (auto &p : contact_tasks_) {
            ContactTask &task = p.second;

            int idx = task.ConstraintForceIndex();
            int dim = task.Dimension();
            // Get constraint forces associate with constraint
            casadi::SX lam = GetDecisionVariables("lam").sym()(
                casadi::Slice(idx, idx + dim));

            // Add any parameters from this task to the dynamics
            casadi::StringVector names =
                utils::casadi::CreateInputNames(task.Function().f());
            casadi::SXVector task_in = GetSymbolicFunctionInput(names);

            // Add any unique inputs to the function input
            AppendFunctionInputs(inames, in, names, task_in);

            // Evaluate Jacobian
            casadi::SX dc = task.Function().f()(task_in)[1];
            casadi::SX J = jacobian(dc, GetParameters("qvel").sym());

            // Get translational component of Jacobian
            casadi::SX Jt = J(casadi::Slice(0, 3), (casadi::Slice(0, nv())));

            // Add joint-space forces associated with the task
            dyn -= mtimes(Jt.T(), lam);
        }

        // Any holonomic constraints
        for (auto &p : holonomic_constraints_) {
            HolonomicConstraint &c = p.second;
            int idx = c.ConstraintForceIndex();
            int dim = c.Dimension();
            // Get constraint forces associate with constraint
            casadi::SX lam = GetDecisionVariables("lam").sym()(
                casadi::Slice(idx, idx + dim));

            // Add any parameters from this task to the dynamics
            casadi::StringVector names =
                utils::casadi::CreateInputNames(c.Function().f());
            casadi::SXVector c_in = GetSymbolicFunctionInput(names);

            // Add any unique inputs to the function input
            AppendFunctionInputs(inames, in, names, c_in);

            // Evaluate Jacobian
            casadi::SX dc = c.Function().f()(c_in)[1];
            casadi::SX J = jacobian(dc, GetParameters("qvel").sym());

            // Add joint-space forces associated with the task
            dyn -= mtimes(J.T(), lam);
        }

        // Add contact forces as inputs
        in.push_back(GetDecisionVariables("lam").sym());
        inames.push_back("lam");

        AddConstraint("dynamics", dyn, in, optimisation::BoundsType::kEquality);

        // ? If projected dynamics, compute null-space dynamics independently
        // ? (i.e. using Eigen)
    }

    // Creates the program for the given conditions
    void UpdateProgramParameters();

    /**
     * @brief Dimension of the configuration variables \f$ q(t) \f$ for the
     * dynamic system.
     *
     * @return const int&
     */
    const int &nq() const { return nq_; }

    /**
     * @brief Dimension of the configuration variables \f$ \dot{q}(t) \f$ for
     * the dynamic system.
     *
     * @return const int&
     */
    const int &nv() const { return nv_; }

   private:
    int nq_ = 0;
    int nv_ = 0;

    casadi::Function fdyn_;

    std::unordered_map<std::string, TrackingTask> tracking_tasks_;
    std::unordered_map<std::string, ContactTask> contact_tasks_;
    std::unordered_map<std::string, HolonomicConstraint> holonomic_constraints_;
};

// TODO - Place this is a utility
Eigen::Quaternion<double> RPYToQuaterion(const double roll, const double pitch,
                                         const double yaw);

}  // namespace control
}  // namespace damotion

#endif /* OSC_OSC_H */
