#ifndef OSC_OSC_H
#define OSC_OSC_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>
#include <map>
#include <string>

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

    // Add constraints

    void AddHolonomicConstraint(damotion::system::HolonomicConstraint &c) {
        // Throw warning if variable inputs are not correct

        // Add parameters to map
    }

    // Task
    class Task {
       public:
        Task() = default;
        ~Task() = default;

        // Evaluates the task and its first and second time derivatives
        eigen::FunctionWrapper x;
        // Parameters
        casadi::SXVector p;

        // Input names
        std::vector<std::string> inames;
        // Output names
        std::vector<std::string> onames;
    };

    // Cost
    class Cost {
       public:
        // Relative cost weighting matrix
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> W;

        Cost() = default;
        ~Cost() = default;

        Cost(casadi::Function &f);

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

    // Associated with tracking a given point in SE(3)
    class TrackingTask : public Task {
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

        /**
         * @brief Updates the tracking error e and de using the current state of
         * the systema and the reference pose xr and/or qr
         *
         */
        void UpdateTrackingError();
    };

    // Associated with point-contact with a given surface
    class ContactTask : public Task {
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

    struct Variables {
        // Number of distinct variable types
        const int sz = 3;
        // Generalised acceleration
        casadi::SX qacc;
        // Control inputs
        casadi::SX ctrl;
        // Contact wrenches
        casadi::SX lambda;
    };

    Variables &GetVariables() {
        if (var_) {
            return *var_;
        } else {
            std::runtime_error(
                "Program has not been initialised, no variables created");
        }
    }

    void AddCost(casadi::Function &c) {
        // Generate cost function and derivatives
    }

    void AddTrackingTask(const std::string &name, casadi::Function &x) {
        // Create new task
        TrackingTask task;
        // Add any parameters to the parameter map
        for (int i = 0; i < x.n_in(); ++i) {
            AddParameters(x.name_in(i), x.size1_in(i));
        }

        // Create translational task
        task.type = TrackingTask::Type::kTranslational;
        // Temporarily add function x to be wrapped
        task.x = eigen::FunctionWrapper(x);
        // Add to map
        tracking_tasks_[name] = task;
    }

    void SetUpTrackingTask() {
        damotion::common::Profiler("osc:setup_tracking_task");

        std::string name;
        TrackingTask &task = tracking_tasks_[name];

        // Determine dimension of the tracking task
        int ndim = 3;
        if (task.type == TrackingTask::Type::kFull) {
            ndim = 6;
        }

        // Get original function
        casadi::Function f = task.x.f();

        // Create inputs
        casadi::SXVector in;
        for (int i = 0; i < f.n_in(); ++i) {
            in.push_back(casadi::SX::sym(f.name_in()[i], f.size1_in(i)));
        }
        // Evaluate function with symbolic inputs
        casadi::SXVector out = f(in);
        // Task Error and PD gains
        Eigen::VectorX<casadi::SX> e, de, xacc;
        Eigen::DiagonalMatrix<casadi::SX, Eigen::Dynamic> Kp(ndim), Kd(ndim);
        eigen::toEigen(out[2], xacc);
        eigen::toEigen(casadi::SX::sym("e", ndim), e);
        eigen::toEigen(casadi::SX::sym("de", ndim), de);
        eigen::toEigen(casadi::SX::sym("Kp", ndim), Kp.diagonal());
        eigen::toEigen(casadi::SX::sym("Kd", ndim), Kd.diagonal());

        // Get necessary components of task acceleration
        if (task.type == TrackingTask::Type::kTranslational) {
            xacc = xacc.topRows(3);
        } else if (task.type == TrackingTask::Type::kRotational) {
            xacc = xacc.bottomRows(3);
        }

        // Create tracking cost
        casadi::SX c = (xacc - (Kp * e + Kd * de)).squaredNorm();

        // Create foramtted input expression
        CreateDefaultFunctionInputs(in, task.inames);
        // Add all parameters
        for (int i = 0; i < task.p.size(); ++i) {
            in.push_back(task.p[i]);
            task.inames.push_back(task.p[i].name());
        }

        // Update function with re-arranged input
        task.onames = {"xpos", "xvel", "xacc"};
        task.x = eigen::FunctionWrapper(
            casadi::Function(f.name(), in, out, task.inames, task.onames));

        // Add tracking-specific parameters that don't need to be added to the
        // parameter map
        std::vector<std::string> name_in = task.inames;
        in.push_back(casadi::SX::sym("e", ndim));
        name_in.push_back("e");
        in.push_back(casadi::SX::sym("de", ndim));
        name_in.push_back("de");
        in.push_back(casadi::SX::sym("Kp", ndim));
        name_in.push_back("Kp");
        in.push_back(casadi::SX::sym("Kd", ndim));
        name_in.push_back("Kd");

        // Create tracking task objective
        out = {c};
        casadi::Function tracking_cost(f.name() + "_tracking_cost", in, out,
                                       name_in, {"c"});
        // Create cost (generates the gradients and Jacobians automatically)
        Cost cost(tracking_cost);

        // Assign vector data to inputs
        task.x.setInput(task.x.f().index_in("qacc"), qacc_);
        task.x.setInput(task.x.f().index_in("ctrl"), qacc_);
        task.x.setInput(task.x.f().index_in("lam"), lambda_);
        for (int i = 0; i < task.p.size(); i++) {
            task.x.setInput(task.x.f().index_in(task.p[i].name()),
                            parameter_map_[task.p[i].name()]);
        }

        // Set inputs for cost function and derivatives
        for (eigen::FunctionWrapper f : {cost.c, cost.g, cost.H}) {
            f.setInput(f.f().index_in("qacc"), qacc_);
            f.setInput(f.f().index_in("ctrl"), qacc_);
            f.setInput(f.f().index_in("lam"), lambda_);
            // Parameters
            for (int i = 0; i < task.p.size(); i++) {
                task.x.setInput(task.x.f().index_in(task.p[i].name()),
                                parameter_map_[task.p[i].name()]);
            }
            // Set tracking inputs
            f.setInput(f.f().index_in("e"), task.e);
            f.setInput(f.f().index_in("de"), task.de);
            f.setInput(f.f().index_in("Kp"), task.Kp.diagonal());
            f.setInput(f.f().index_in("Kd"), task.Kd.diagonal());
        }

        // Add cost to cost map
        // costs_["TEST"] = cost; // ! Fix
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
        }
    }

    void AddParameters(const std::string &name, int sz) {
        damotion::common::Profiler("osc:add_parameters");
        // If its a variable name, ignore it
        if (name == "qacc" || name == "ctrl" || name == "lam") {
            return;
        }

        // Look up parameters in parameter map and set values
        auto p = parameter_map_.find(name);
        // If doesn't exist, add parameter
        if (p == parameter_map_.end()) {
            parameter_map_[name] = Eigen::VectorXd::Zero(sz);
        } else {
            // ! Throw warning that it already exists?
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

    // Given all information for the problem, initialise the program
    void Initialise() {
        // Determine how many contact wrenches need to be considered
        int n = 0;
        for (auto &task : contact_tasks_) {
            n += task.second.xr.size();  // ! Fix this
        }

        var_ = std::make_unique<Variables>();
        var_->qacc = casadi::SX::sym("qacc");
        var_->ctrl = casadi::SX::sym("ctrl");
        var_->lambda = casadi::SX::sym("lam");
    }

    // Creates the program for the given conditions
    void CreateProgram();

    // Return program data that can then be solved

   private:
    // std::vector<> Cost expressions/functions?
    // std::vector<> Constraint expressions/functions?

    void CreateDefaultFunctionInputs(casadi::SXVector &in,
                                     casadi::StringVector &name_in) {
        in = {};
        name_in = {};
        in.push_back(GetVariables().qacc);
        name_in.push_back("qacc");
        in.push_back(GetVariables().ctrl);
        name_in.push_back("ctrl");
        in.push_back(GetVariables().lambda);
        name_in.push_back("lambda");
    }

    std::unique_ptr<Variables> var_ = nullptr;

    struct TaskPriority {
        // Relative weighting
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> w;
    };

    Eigen::VectorXd qacc_;
    Eigen::VectorXd ctrl_;
    Eigen::VectorXd lambda_;

    std::unordered_map<std::string, TrackingTask> tracking_tasks_;
    std::unordered_map<std::string, ContactTask> contact_tasks_;
    std::unordered_map<std::string, Cost> costs_;

    // Parameter map
    std::unordered_map<std::string, Eigen::VectorXd> parameter_map_;
};

// TODO - Place this is a utility
Eigen::Quaternion<double> RPYToQuaterion(const double roll, const double pitch,
                                         const double yaw);
#endif /* OSC_OSC_H */
