#ifndef OSC_OSC_H
#define OSC_OSC_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>
#include <map>
#include <string>

#include "casadi_utils/codegen.h"
#include "casadi_utils/eigen_wrapper.h"

// ? create functions that compute the necessary constraints and stuff for a
// ? problem, then pass it to a given solver?

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
        eigen::FunctionWrapper x;
        eigen::FunctionWrapper J;

        // Parameters
        casadi::SXVector px;
        casadi::SXVector pJ;
    };

    // Cost
    class Cost {
       public:
        // Cost
        eigen::FunctionWrapper c;
        // Gradient
        eigen::FunctionWrapper g;
        // Hessian
        eigen::FunctionWrapper H;
    };

    // Associated with tracking a given point in SE(3)
    class TrackingTask : public Task {
       public:
        enum class Type { kTranslational, kRotational, kFull };

        // Task
        eigen::FunctionWrapper x;
        // Task tracking cost
        eigen::FunctionWrapper c;

        eigen::FunctionWrapper J;

        void ComputeError();

        Type type;
        // Tracking gains
        Eigen::DiagonalMatrix<double> Kp;
        // Tracking gains
        Eigen::DiagonalMatrix<double> Kd;

        // Error in pose
        Eigen::VectorXd e;
        Eigen::VectorXd de;

        // Desired pose
        Eigen::Vector3d xr;
        Eigen::Quaterniond qr;

        // Expressions for the tracking task x
        casadi::SX xpos;
        casadi::SX xvel;
        casadi::SX xacc;
        casadi::SX J;
    };

    // Associated with point-contact with a given surface
    class ContactTask : public Task {
       public:
        bool inContact = false;
        Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();

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

    void AddTrackingTask(const std::string &name, casadi::Function &x,
                         casadi::Function &J) {
        // Create new task
        TrackingTask task;
        // Add any parameters to the parameter map
        for (int i = 0; i < x.n_in(); ++i) {
            AddParameters(x.name_in(i), x.size1_in(i));
        }
        // Create inputs
        casadi::SXVector in;
        for (int i = 0; i < x.n_in(); ++i) {
            in.push_back(casadi::SX::sym(x.name_in[i], x.size1_in[i]));
        }
        // Evaluate and store expressions
        casadi::SXVector out = x(in);
        task.xpos = out[0];
        task.xvel = out[1];
        task.xacc = out[2];

        // Evaluate Jacobian
        in = {};
        for (int i = 0; i < x.n_in(); ++i) {
            in.push_back(casadi::SX::sym(J.name_in[i], J.size1_in[i]));
        }
        // Evaluate and store expressions
        out = J(in);
        task.J = out[0];

        // Create translational task
        task.type = TrackingTask::Type::kTranslational;
        // Add to map
        tracking_tasks_[name] = task;
    }

    void SetUpTrackingTask() {
        std::string name;
        TrackingTask &task = tracking_tasks_[name];

        int ndim = 3;
        if (task.type == TrackingTask::Type::kFull) {
            ndim = 6;
        }

        // Evaluate constraints
        Eigen::VectorX<casadi::SX> e, de;
        // Task PD gains
        Eigen::Vector3<casadi::SX> Kp, Kd;
        eigen::toEigen(casadi::SX::sym("Kp", ndim), Kp.diagonal());
        eigen::toEigen(casadi::SX::sym("Kd", ndim), Kd.diagonal());
        eigen::toEigen(casadi::SX::sym("e", ndim), e);
        eigen::toEigen(casadi::SX::sym("de", ndim), de);

        // Create function for tracking task
        casadi::SX c = (xacc - (Kp * e + Kd * de)).squaredNorm();

        // ! Create cost from this expression

        // Create functions and set destinations
        casadi::Function xx("", in, out, name_in, name_out);

        // Function inputs and outputs
        casadi::SXVector in, out;
        casadi::StringVector name_in, name_out;
        // Create formatted function inputs
        CreateDefaultFunctionInputs(in, name_in);
        // Add any additional parameters
        for (int i = 0; i < task.px.size(); ++i) {
            in.push_back(task.p[i]);
            name_in.push_back(task.p[i].name());
        }
        // Create parameters and set them
        in.push_back(casadi::SX::sym("e", ndim));
        in.push_back(casadi::SX::sym("de", ndim));
        in.push_back(casadi::SX::sym("Kp", ndim));
        in.push_back(casadi::SX::sym("Kd", ndim));

        // ! Determine if codegen is required
        task.x = eigen::FunctionWrapper(casadi_utils::codegen(xx));

        // Setup inputs to function
        task.x.setInput(0, qacc_);
        task.x.setInput(1, ctrl_);
        task.x.setInput(2, lambda_);
        for (int i = 0; i < task.px.size(); i++) {
            task.x.setInput(3 + i, parameter_map_[task.p[i].name()]);
        }
        // Set tracking inputs
        int n = 3 + task.px.size();
        task.x.setInput(n, task.e);
        task.x.setInput(n + 1, task.de);
        task.x.setInput(n + 2, task.Kp.diagonal());
        task.x.setInput(n + 3, task.Kd.diagonal());
    }

    class HolonomicConstraint {};

    // Generic cost that can be added
    class Cost {};

    // Functions for setting new weightings?
    // Access through the map?
    void UpdateTrackingCostGains(std::string &name,
                                 const Eigen::DiagonalMatrix<double> &Kp,
                                 const Eigen::DiagonalMatrix<double> &Kd) {
        ee_[name].Kp = Kp;
        ee_[name].Kd = Kd;
    }

    void UpdateParameters(eigen::FunctionWrapper &f) {
        // Ignore variable inputs (qacc, ctrl, lambda)
        for (int i = 3; i < f.f().n_in(); ++i) {
            // Look up parameters in parameter map and set values
            auto p = parameter_map_.find(f.f().name_in(i));
            // If it exists, update parameter
            if (p != parameter_map_.end()) {
                f.setInput(i, p->second);
            } else {
                // ! Throw error that parameter is not included
            }
        }
    }

    void AddParameters(const std::string &name, int sz) {
        // If its a variable name, ignore it
        if (name == "qacc" || name == "ctrl" || name == "lam") {
            return;
        }

        // Look up parameters in parameter map and set values
        auto p = parameter_map_.find(name);
        // If it exists, update parameter
        if (p != parameter_map_.end()) {
            parameter_map_[name] = Eigen::VectorXd::Zeros(sz);
        }
    }

    // Given all information for the problem, initialise the program
    void Initialise() {
        // Determine how many contact wrenches need to be considered
        int n = 0;
        for (auto &ee : ee_) {
            n += ee.second.xr.size()  // ! Fix this
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
        Eigen::DiagonalMatrix<double> w;
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
