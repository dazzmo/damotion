#ifndef OSC_OSC_H
#define OSC_OSC_H

#include <Eigen/Core>
#include <algorithm>
#include <casadi/casadi.hpp>
#include <map>
#include <string>

#include "solvers/program.h"
#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"

// ? create functions that compute the necessary constraints and stuff for a
// ? problem, then pass it to a given solver?

#include "common/profiler.h"
#include "system/constraint.h"

// Variables at play: qacc, u, lambda

using namespace casadi_utils;

namespace damotion {
namespace control {

class OSCController : public solvers::Program {
   public:
    OSCController() = default;
    ~OSCController() = default;

    OSCController(int nq, int nv, int nu) {
        // Add default variables and parameters to map
        AddVariables("qacc", nv);
        AddVariables("ctrl", nu);
        AddVariables("lam", 0);

        AddParameters("qpos", nq);
        AddParameters("qvel", nv);
    }

    // Holonomic Constraint
    class HolonomicConstraint : public Constraint {
       public:
        HolonomicConstraint() = default;
        ~HolonomicConstraint() = default;

        // Constraint forces if using lagrangian principles
        int lam_idx = 0;

        // Constraint
        casadi::SX c;
        // Constraint first derivative
        casadi::SX dc_dt;
        // Constraint second derivative
        casadi::SX d2c_dt2;
        // Constraint jacobian
        casadi::SX J;
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

    void CreateHolonomicConstraint(const std::string &name, casadi::Function &c,
                                   HolonomicConstraint &constraint) {
        for (int i = 0; i < c.n_in(); ++i) {
            // Add inputs to constraint
            constraint.in.push_back(
                casadi::SX::sym(c.name_in(i), c.size1_in(i)));
            constraint.inames.push_back(c.name_in(i));

            // Add any parameters to the parameter map
            if (!IsVariable(c.name_in(i))) {
                AddParameters(c.name_in(i), c.size1_in(i));
            }
        }

        // Evaluate symbolic outputs
        casadi::SXVector out = c(constraint.in);
        constraint.c = out[0];
        constraint.dc_dt = out[1];
        constraint.d2c_dt2 = out[2];
        constraint.J =
            jacobian(constraint.dc_dt,
                     casadi::SX::sym("qvel", c.size1_in(c.index_in("qvel"))));
    }

    void AddTrackingTask(const std::string &name, casadi::Function &x,
                         const TrackingTask::Type &type) {
        // Create new task
        TrackingTask task;
        CreateHolonomicConstraint(name, x, task);
        // Add tracking task to map
        tracking_tasks_[name] = task;
    }

    void AddHolonomicConstraint(const std::string &name, casadi::Function &c) {
        // Create new constraint
        HolonomicConstraint constraint;
        CreateHolonomicConstraint(name, c, constraint);
        // Add holonomic constraint to map
        holonomic_constraints_[name] = constraint;
    }

    void AddDynamics(casadi::Function &f) { fdyn_ = f; }

    void RegisterTrackingTask(TrackingTask &task) {
        damotion::common::Profiler("OSCController::RegisterTrackingTask");

        // Add new parameter for program
        int dim = 6;
        if (task.type == TrackingTask::Type::kRotational ||
            task.type == TrackingTask::Type::kTranslational) {
            dim = 3;
        }
        // Add tracking acceleration as a parameter to the program
        AddParameters(task.name() + "_xacc_d", dim);

        // Task Error and PD gains
        casadi::SX xacc_d = casadi::SX::sym(task.name() + "_xacc_d");
        Eigen::VectorX<casadi::SX> xacc_e, xacc_d_e;
        eigen::toEigen(task.d2c_dt2, xacc_e);
        eigen::toEigen(xacc_d, xacc_d_e);

        // Get necessary components of task acceleration
        if (task.type == TrackingTask::Type::kTranslational) {
            xacc_e = xacc_e.topRows(3);
        } else if (task.type == TrackingTask::Type::kRotational) {
            xacc_e = xacc_e.bottomRows(3);
        }

        // Create tracking cost
        casadi::SX obj = (xacc_e - xacc_d_e).squaredNorm();

        casadi::SXVector in = task.in, out;
        std::vector<std::string> inames = task.inames;

        in.push_back(xacc_d);
        inames.push_back(task.name() + "_xacc_d");

        // Create tracking objective
        casadi::Function tracking_cost(task.name() + "_tracking_cost", in,
                                       {obj}, inames, {"obj"});

        // Create cost and derivatives given the optimisation vector x
        Program::Cost cost(task.name() + "_tracking_cost", tracking_cost,
                           DecisionVariableVector());

        // Register function data to task
        SetFunctionData(task.con);

        // Add tracking cost to program
        AddCost(task.name() + "_tracking_cost", cost);
    }

    void RegisterHolonomicConstraint(HolonomicConstraint &con) {
        damotion::common::Profiler(
            "OSCController::RegisterHolonomicConstraint");

        casadi::SXVector in = con.in;
        std::vector<std::string> inames = con.inames;

        // Set second rate of change of constraint to zero to ensure no change
        casadi::Function constraint(con.name(), in, {con.d2c_dt2}, inames,
                                    {con.name() + "_d2c_dt2"});
        // Create cost based
        Constraint c(con.name(), constraint, DecisionVariableVector());
        // Add constraint to program
        AddConstraint(con.name(), c);
    }

    // Functions for setting new weightings?
    // Access through the map?
    void UpdateTrackingCostGains(
        std::string &name,
        const Eigen::DiagonalMatrix<double, Eigen::Dynamic> &Kp,
        const Eigen::DiagonalMatrix<double, Eigen::Dynamic> &Kd) {
        // ! Throw error if name is not in lookup
        tracking_tasks_[name].Kp = Kp;
        tracking_tasks_[name].Kd = Kd;
    }

    // Given all information for the problem, initialise the program
    void Initialise() {
        // Determine how many contact wrenches need to be considered
        int nc = 0;
        for (auto &task : contact_tasks_) {
            // Assign index to task
            task.second.lam_idx = nc;
            nc += task.second.dim();  // ! Fix this
        }
        // Add any holonomic constraints
        // TODO - Make projection a possibility
        for (auto &con : holonomic_constraints_) {
            // Assign index
            con.second.lam_idx = nc;
            nc += con.second.dim();  // ! Fix this
        }

        // Resize lambda to account for all constraint forces
        GetVariables("lam") = casadi::SX::sym("lam", nc);

        // Create optimisation vector with given ordering
        ConstructDecisionVariableVector({"qacc", "ctrl", "lam"});

        // Create contact constraints
        for (auto &p : contact_tasks_) {
            HolonomicConstraint &task = p.second;

            // No-slip condition
            RegisterHolonomicConstraint(task);

            // Get contact forces from force vector
            casadi::SX lam =
                GetVariables("lam")(casadi::Slice(task.lam_idx, 3));

            // Add friction constraint
            Constraint friction =
                FrictionConeConstraint(task.name() + "_friction", lam);
            AddConstraint(task.name() + "_friction", friction);
        }

        // Add any tracking tasks
        for (auto &task : tracking_tasks_) {
            RegisterTrackingTask(task.second);
        }

        // Add any other holonomic constraints
        for (auto &con : holonomic_constraints_) {
            RegisterHolonomicConstraint(con.second);
        }

        // Create constrained dynamics constraint
        ComputeConstrainedDynamics();
    }

    /**
     * @brief Create a linearised friction cone with tuneable parameter mu to
     * adjust the friction cone constraints
     *
     * @param lambda
     */
    Constraint FrictionConeConstraint(const std::string &name,
                                      casadi::SX &lambda) {
        // Square pyramid approximation
        casadi::SX l_x = lambda(0), l_y = lambda(1), l_z = lambda(2);

        casadi::SX mu = casadi::SX::sym(name + "mu");

        AddParameters(name + "_friction_mu", 1);

        // Friction cone constraint with square pyramid approximation
        casadi::SX cone(4);
        cone(0) = sqrt(2.0) * l_x + mu * l_z;
        cone(1) = -sqrt(2.0) * l_x - mu * l_z;
        cone(2) = sqrt(2.0) * l_y + mu * l_z;
        cone(3) = -sqrt(2.0) * l_y - mu * l_z;

        // Create function
        casadi::Function fcone(name + "_friction",
                               {GetVariables("qacc"), GetVariables("ctrl"),
                                GetVariables("lam"), mu},
                               {cone}, {"qacc", "ctrl", "lam", "mu"},
                               {"friction"});

        Constraint c_cone(name, fcone, DecisionVariableVector());
        // Set constraint to be positive
        c_cone.lb().setZero();
        c_cone.ub().setConstant(1e8);

        return c_cone;
    }

    void UpdateContactFrictionCoefficient(std::string &name, const double &mu) {
        // ! Throw error if name is not in lookup
        Eigen::VectorXd val(1);
        val[0] = mu;
        SetParameter(name + "_friction_mu", val);
    }

    void ComputeConstrainedDynamics() {
        // Get inputs for function
        casadi::SXVector in;
        for (int i = 0; i < fdyn_.n_in(); ++i) {
            in.push_back(
                casadi::SX::sym(fdyn_.name_in()[i], fdyn_.size1_in(i)));
        }
        // Evaluate dynamics
        casadi::SX dyn = fdyn_(in)[0];

        std::vector<std::string> inames = {};

        // Any contact-tasks
        for (auto &p : contact_tasks_) {
            // Get contact task
            ContactTask &task = p.second;

            // Add any parameters from this task to the dynamics
            for (std::string &p : task.inames) {
                // Add any parameters that aren't in the function
                if (std::find(inames.begin(), inames.end(), p) ==
                    inames.end()) {
                    /* v does not contain x */
                    inames.push_back(p);
                    in.push_back(casadi::SX::sym(p, task.con.f().size1_in(p)));
                }
            }

            casadi::SX &J = task.J;
            casadi::SX lam =
                GetVariables("lam")(casadi::Slice(task.lam_idx, task.dim()));
            // Add joint-space forces from lambda
            dyn -= mtimes(J.T(), lam);
        }

        // Any holonomic constraints
        for (auto &p : holonomic_constraints_) {
            // Get contact task
            HolonomicConstraint &constraint = p.second;

            // Add any parameters for this task to the dynamics
            for (std::string &p : constraint.inames) {
                // Add any parameters that aren't in the function
                if (std::find(inames.begin(), inames.end(), p) ==
                    inames.end()) {
                    /* v does not contain x */
                    inames.push_back(p);
                    // ! in.push_back(casadi::SX::sym(p,
                    // constraint.c.f().size1_in(p)));
                }
            }

            casadi::SX &J = constraint.J;
            casadi::SX lam = GetVariables("lam")(
                casadi::Slice(constraint.lam_idx, constraint.dim()));
            // Add joint-space forces from lambda
            dyn -= mtimes(J.T(), lam);
        }

        // Create function for dynamics
        fdyn_ = casadi::Function("dynamics", in, {dyn}, inames, {"dyn"});

        // Create constraint from the function and add it to the program
        Constraint c("dynamics", fdyn_, DecisionVariableVector());
        AddConstraint("dynamics", c);

        // ? If projected dynamics, compute null-space dynamics independently
        // ? (i.e. using Eigen)
    }

    // Creates the program for the given conditions
    void UpdateProgramParameters();

   private:
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
