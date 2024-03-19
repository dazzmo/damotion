#ifndef OSC_OSC_H
#define OSC_OSC_H

#include <Eigen/Core>
#include <algorithm>
#include <casadi/casadi.hpp>
#include <map>
#include <string>

#include "control/quad_prog.h"
#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"

// ? create functions that compute the necessary constraints and stuff for a
// ? problem, then pass it to a given solver?

#include "common/profiler.h"
#include "system/constraint.h"

// Variables at play: qacc, u, lambda

using namespace casadi_utils;

class OSCController {
   public:
    OSCController() = default;
    ~OSCController() = default;

    OSCController(int nq, int nv, int nu) {
        // Add variables to map
        AddVariables("qacc", nv);
        AddVariables("ctrl", nu);
        AddVariables("lam", 0);
    }

    // Holonomic Constraint
    class HolonomicConstraint {
       public:
        HolonomicConstraint() = default;
        ~HolonomicConstraint() = default;

        // Name of the constraint
        std::string name = "";

        // Inputs
        casadi::SXVector in;

        // Dimension of the constraint
        int dim = 0;

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

        // Function to compute constraint and rates of change
        eigen::FunctionWrapper f;
        // Function to compute constraint jacobian
        eigen::FunctionWrapper jac;

        // Input names
        std::vector<std::string> inames;
        // Output names
        std::vector<std::string> onames;
    };

    // Cost
    class Cost {
       public:
        // Relative cost weighting matrix
        double w;

        Cost() = default;
        ~Cost() = default;

        Cost(casadi::Function &f, const casadi::SX &x);

        // Cost
        eigen::FunctionWrapper c;
        // Gradient
        eigen::FunctionWrapper g;
        // Hessian
        eigen::FunctionWrapper H;

        // Input names
        std::vector<std::string> inames;
        // Output names
        std::vector<std::string> onames;
    };

    class Constraint {
       public:
        Constraint() = default;
        ~Constraint() = default;

        Constraint(casadi::Function &f, const casadi::SX &x);

        int dim = 0;

        // Constraint
        eigen::FunctionWrapper c;
        // Jacobian
        eigen::FunctionWrapper jac;

        Eigen::VectorXd lb;
        Eigen::VectorXd ub;

        // Input names
        std::vector<std::string> inames;
        // Output names
        std::vector<std::string> onames;
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
        // Add any parameters to the parameter map
        for (int i = 0; i < c.n_in(); ++i) {
            // Add input
            constraint.in.push_back(
                casadi::SX::sym(c.name_in(i), c.size1_in(i)));
            constraint.inames.push_back(c.name_in(i));
            // Add to parameter map
            if (!IsVariable(c.name_in(i))) {
                AddParameters(c.name_in(i), c.size1_in(i));
            }
        }

        // Determine dimension
        constraint.dim = c.size1_out(0);

        // Compute symbolic outputs
        casadi::SXVector out = c(constraint.in);
        constraint.c = out[0];
        constraint.dc_dt = out[1];
        constraint.d2c_dt2 = out[2];
        constraint.J =
            jacobian(constraint.dc_dt,
                     casadi::SX::sym("qvel", c.size1_in(c.index_in("qvel"))));

        // Wrap function and add to map
        constraint.f = eigen::FunctionWrapper(c);
    }

    void AddTrackingTask(const std::string &name, casadi::Function &x,
                         const TrackingTask::Type &type) {
        // Create new task
        TrackingTask task;
        CreateHolonomicConstraint(name, x, task);

        // Determine task type and dimension
        task.type = type;
        if (type == TrackingTask::Type::kRotational ||
            type == TrackingTask::Type::kTranslational) {
            task.dim = 3;
        } else {
            task.dim = 6;
        }

        // Add tracking task to map
        tracking_tasks_[name] = task;
    }

    void AddHolonomicConstraint(const std::string &name, casadi::Function &c) {
        // Create new constraint
        HolonomicConstraint constraint;
        CreateHolonomicConstraint(name, c, constraint);
        // Add to map
        holonomic_constraints_[name] = constraint;
    }

    void AddDynamics(casadi::Function &f) { fdyn_ = f; }

    void SetUpTrackingTask(TrackingTask &task) {
        damotion::common::Profiler("osc:setup_tracking_task");

        // Add new parameter for program
        AddParameters(task.name + "_xacc_d", task.dim);

        // Task Error and PD gains
        casadi::SX xacc_d = casadi::SX::sym(task.name + "_xacc_d");
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
        inames.push_back(task.name + "_xacc_d");

        // Create tracking objective
        casadi::Function tracking_cost(task.name + "_tracking_cost", in, {obj},
                                       inames, {"obj"});
                                       
        // Create cost and derivatives given the optimisation vector x
        Cost cost(tracking_cost, x_);

        // Assign vector data to inputs
        SetFunctionData(task.f);

        SetFunctionData(cost.c);
        SetFunctionData(cost.g);
        SetFunctionData(cost.H);

        // Add associated cost to cost map
        costs_[task.name + "_tracking_cost"] = cost;
    }

    void SetUpHolonomicConstraint(HolonomicConstraint &con) {
        damotion::common::Profiler("OSCController::SetUpHolonomicConstraint");

        // Add tracking-specific parameters that don't need to be added to the
        // parameter map
        casadi::SXVector in = con.in, out;
        std::vector<std::string> inames = con.inames;

        // Set second rate of change of constraint to zero to ensure no change
        casadi::Function constraint(con.name + "_constraint", in, {con.d2c_dt2},
                                    inames, {con.name + "_d2c_dt2"});
        // Create cost based
        Constraint c(constraint, x_);

        // Assign vector data to inputs
        SetFunctionData(c.c);
        SetFunctionData(c.jac);

        // Add associated cost to cost map
        constraints_[con.name + "_constraint"] = c;
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

    void SetParameter(const std::string &name, const Eigen::VectorXd &val) {
        // Look up parameters in parameter map and set values
        auto p = parameter_map_.find(name);
        // If it exists, update parameter
        if (p != parameter_map_.end()) {
            p->second = val;
        } else {
            // ! Throw error that parameter is not included
            std::cout << "Parameter " << name
                      << "is not in the parameter map!\n";
        }
    }

    void AddParameters(const std::string &name, int sz) {
        damotion::common::Profiler("OSCController::AddParameters");
        // Look up parameters in parameter map and set values
        auto p = parameter_map_.find(name);
        // If doesn't exist, add parameter
        if (p == parameter_map_.end()) {
            parameter_map_[name] = Eigen::VectorXd::Zero(sz);
        }
    }

    void AddVariables(const std::string &name, int sz) {
        damotion::common::Profiler("OSCController::AddVariables");
        // Look up parameters in parameter map and set values
        auto p = variables_.find(name);
        // If doesn't exist, add parameter
        if (p == variables_.end()) {
            variables_[name] = casadi::SX::sym(name, sz);
        }
    }

    /**
     * @brief Prints the current set of parameters for the controller to the
     * screen
     *
     */
    void ListParameters() {
        std::cout << "Parameter\tSize\n";
        for (auto p : parameter_map_) {
            std::cout << p.first << "\t" << p.second.size() << " x 1\n";
        }
    }

    /**
     * @brief Prints the current set of parameters for the controller to the
     * screen
     *
     */
    void ListVariables() {
        std::cout << "Variable\tSize\n";
        for (auto v : variables_) {
            std::cout << v.first << "\t" << v.second.size() << " x 1\n";
        }
    }

    // Given all information for the problem, initialise the program
    void Initialise() {
        // Determine how many contact wrenches need to be considered
        int nc = 0;
        for (auto &task : contact_tasks_) {
            // Assign index to task
            task.second.lam_idx = nc;
            nc += task.second.xr.size();  // ! Fix this
        }
        // Add any holonomic constraints
        // TODO - Make projection a possibility
        for (auto &con : holonomic_constraints_) {
            // Assign index
            con.second.lam_idx = nc;
            nc += con.second.dim;  // ! Fix this
        }

        // Resize lambda
        variables_["lam"].resize(nc, 1);

        // Create optimisation vector with given ordering
        CreateOptimisationVector({"qacc", "ctrl", "lam"});

        // Create contact constraints
        for (auto &task : contact_tasks_) {
            // Get contact forces from force vector
            casadi::SX lam = var_->lam(casadi::Slice(task.second.lam_idx, 3));
            constraints_[task.first + "_friction"] =
                FrictionConeConstraint(task.first, lam);
        }

        // Add any tracking tasks
        for (auto &task : tracking_tasks_) {
            SetUpTrackingTask(task.second);
        }

        // Add any holonomic constraints
        for (auto &con : holonomic_constraints_) {
            SetUpHolonomicConstraint(con.second);
        }

        // Create dynamics subject to holonomic constraints
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
        casadi::Function fcone(
            name + "_friction",
            {variables_["qacc"], variables_["ctrl"], variables_["lam"], mu},
            {cone}, {"qacc", "ctrl", "lam", "mu"}, {"friction"});

        Constraint c_cone(fcone, x_);
        // Set constraint to be positive
        c_cone.lb.setZero();
        c_cone.ub.setConstant(1e8);

        return c_cone;
    }

    void UpdateContactFrictionCoefficient(std::string &name, const double &mu) {
        // ! Throw error if name is not in lookup
        Eigen::VectorXd val(1);
        val[0] = mu;
        SetParameter(name + "_friction_mu", val);
    }

    void ComputeConstrainedDynamics() {
        // Get unconstrained dynamics M qacc + C qvel + G - B u = 0

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

            // Add any parameters for this task to the dynamics
            for (std::string &p : task.inames) {
                // Add any parameters that aren't in the function
                if (std::find(inames.begin(), inames.end(), p) ==
                    inames.end()) {
                    /* v does not contain x */
                    inames.push_back(p);
                    in.push_back(casadi::SX::sym(p, task.f.f().size1_in(p)));
                }
            }

            casadi::SX &J = task.J;
            casadi::SX lam =
                variables_["lam"](casadi::Slice(task.lam_idx, task.dim));
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
            casadi::SX lam = variables_["lam"](
                casadi::Slice(constraint.lam_idx, constraint.dim));
            // Add joint-space forces from lambda
            dyn -= mtimes(J.T(), lam);
        }

        // Create function for dynamics
        fdyn_ = casadi::Function("dynamics", in, {dyn}, inames, {"dyn"});

        // Create constraint for problem
        Constraint c(fdyn_, x_);

        // Add constraints to map
        constraints_["dynamics"] = c;

        // ? If projected dynamics, compute null-space dynamics independently
        // ? (i.e. using Eigen)
    }

    // Creates the program for the given conditions
    void CreateProgram();

    void CreateOptimisationVector(const std::vector<std::string> &variables) {
        casadi::SXVector x;
        int idx = 0;
        for (int i = 0; i < variables.size(); ++i) {
            x.push_back(variables_[variables[i]]);
            variable_idx_[variables[i]] = idx;
            idx += variables_[variables[i]].size1();
        }

        // Create optimisation vector
        x_ = casadi::SX::vertcat(x);
    }

    // Return program data that can then be solved

    /**
     * @brief Indicates whether the provided variable is associated with the
     * optimisation variables for the problem
     *
     * @param name
     * @return true
     * @return false
     */
    inline bool IsVariable(const std::string &name) {
        // Check if it hits in the variable map
        auto p = variables_.find(name);
        // Indicate if it is present in the variable map
        return p != variables_.end();
    }

    /**
     * @brief Indicates whether the provided variable is associated with the
     * optimisation variables for the problem
     *
     * @param name
     * @return true
     * @return false
     */
    inline bool IsParameter(const std::string &name) {
        // Check if it hits in the variable map
        auto p = parameter_map_.find(name);
        // Indicate if it is present in the variable map
        return p != parameter_map_.end();
    }

    void SetFunctionData(eigen::FunctionWrapper &f) {
        // Go through all function inputs and set both variable and parameter
        // locations
        for (int i = 0; i < f.f().n_in(); ++i) {
            std::string name = f.f().name_in(i);
            if (!IsVariable(name)) {
                f.setInput(f.f().index_in(name), parameter_map_[name]);
            } else {
                f.setInput(f.f().index_in(name), variable_map_[name]);
            }
        }
    }

   private:
    casadi::Function fdyn_;

    // Optimisation vector
    casadi::SX x_;

    std::unordered_map<std::string, TrackingTask> tracking_tasks_;
    std::unordered_map<std::string, ContactTask> contact_tasks_;
    std::unordered_map<std::string, HolonomicConstraint> holonomic_constraints_;

    // Program data

    // Variables
    std::unordered_map<std::string, casadi::SX> variables_;
    // Variable indices
    std::unordered_map<std::string, int> variable_idx_;

    // Constraints
    std::unordered_map<std::string, Constraint> constraints_;
    // Costs
    std::unordered_map<std::string, Cost> costs_;

    std::unordered_map<std::string, Eigen::VectorXd> variable_map_;
    // Parameters
    std::unordered_map<std::string, Eigen::VectorXd> parameter_map_;
};

// TODO - Place this is a utility
Eigen::Quaternion<double> RPYToQuaterion(const double roll, const double pitch,
                                         const double yaw);
#endif /* OSC_OSC_H */
