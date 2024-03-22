#ifndef SOLVERS_PROGRAM_H
#define SOLVERS_PROGRAM_H

#include <casadi/casadi.hpp>

#include "common/profiler.h"
#include "utils/casadi.h"
#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"

#include "solvers/cost.h"
#include "solvers/constraint.h"

using namespace casadi_utils;

namespace damotion {
namespace optimisation {

class Program {
   public:
    Program() = default;
    ~Program() = default;

    Program(const std::string &name) : name_(name) {}

    
    void AddVariables(const std::string &name, const int sz) {
        if (variables_.find(name) == variables_.end()) {
            variables_[name] = casadi::SX::sym(name, sz);
            variable_map_[name] = Eigen::VectorXd::Zero(sz);
        } else {
            // ! Make this more sophisticated
            std::cout << "Variables with name " << name << " already added!\n";
        }
    }

    /**
     * @brief Returns the symbolic vector for the variable given by name
     *
     * @param name
     * @return casadi::SX&
     */
    casadi::SX &GetVariables(const std::string &name) {
        if (!IsVariable(name)) {
            throw std::runtime_error("Variable " + name +
                                     "is not included in this program");
        } else {
            return variables_[name];
        }
    }

    /**
     * @brief The starting index of the variables given by name within the
     * DecisionVaraibleVector() x
     *
     * @param name
     * @return int
     */
    int GetVariableIndex(const std::string &name) {
        if (!IsVariable(name)) {
            std::cout << "Variable " << name
                      << " is not within this program!\n";
            return -1;
        } else {
            return variable_idx_[name];
        }
    }

    /**
     * @brief Returns the symbolic vector for the parameter given by name
     *
     * @param name
     * @return casadi::SX&
     */
    casadi::SX &GetParameters(const std::string &name) {
        if (!IsParameter(name)) {
            throw std::runtime_error("Parameter " + name +
                                     "is not included in this program");
        } else {
            return parameters_[name];
        }
    }

    Eigen::Ref<Eigen::VectorXd> GetVariableUpperBounds(
        const std::string &name) {
        if (IsVariable(name)) {
            return ubx_.middleRows(variable_idx_[name],
                                   variable_map_[name].size());
        } else {
            throw std::runtime_error(name +
                                     "is not a variable in this program");
        }
    }

    Eigen::Ref<Eigen::VectorXd> GetVariableLowerBounds(
        const std::string &name) {
        if (IsVariable(name)) {
            return lbx_.middleRows(variable_idx_[name],
                                   variable_map_[name].size());
        } else {
            throw std::runtime_error(name +
                                     "is not a variable in this program");
        }
    }

    /**
     * @brief Indicates whether the provided variable is associated with the
     * optimisation variables for the problem
     *
     * @param name
     * @return true
     * @return false
     */
    inline bool IsVariable(const std::string &name) {
        // Indicate if it is present in the variable map
        return variables_.find(name) != variables_.end();
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
        // Indicate if it is present in the parameter map
        return parameter_map_.find(name) != parameter_map_.end();
    }

    void AddCost(const std::string &name, casadi::SX &cost,
                 casadi::SXVector &in);

    void RegisterCost(const Cost &cost);
    void RegisterCosts();
    Cost &GetCost(const std::string &name) { return costs_[name]; }

    void AddConstraint(const std::string &name, casadi::SX &constraint,
                       casadi::SXVector &in);
    void RegisterConstraint(const Constraint &constraint);
    void RegisterConstraints();
    Constraint &GetConstraint(const std::string &name) {
        return constraints_[name];
    }

    void ConstructConstraintVector() {
        // TODO - Make custom ordering possible
        nc_ = 0;
        for (auto &p : constraints_) {
            Constraint &c = p.second;
            c.SetIndex(nc_);
            nc_ += c.dim();
        }

        // Create constraint bounds
        lbg_.resize(nc_);
        ubg_.resize(nc_);

        lbg_.setConstant(-std::numeric_limits<double>::infinity());
        ubg_.setConstant(std::numeric_limits<double>::infinity());

        // Set bounds
        for (auto &p : constraints_) {
            Constraint &c = p.second;
            lbg_.middleRows(c.idx(), c.dim()) = c.lb();
            ubg_.middleRows(c.idx(), c.dim()) = c.ub();
        }
    }

    void SetFunctionData(utils::casadi::FunctionWrapper &f) {
        // Go through all function inputs and set both variable and parameter
        // locations
        for (int i = 0; i < f.f().n_in(); ++i) {
            std::string name = f.f().name_in(i);
            if (IsVariable(name)) {
                f.setInput(f.f().index_in(name), variable_map_[name]);
            } else if (IsParameter(name)) {
                f.setInput(f.f().index_in(name), parameter_map_[name]);
            } else {
                std::cout << "Input " << name
                          << " is not registered in program!\n";
            }
        }
    }

    void UpdateDecisionVariables(const Eigen::VectorXd &x) {
        damotion::common::Profiler profiler("Program::UpdateDecisionVariables");
        // Use current mapping to compute variables
        for (auto &xi : variables_) {
            variable_map_[xi.first] = x.middleRows(
                variable_idx_[xi.first], variable_map_[xi.first].size());
        }
    }

    void ResizeDecisionVariables(const std::string &name, const int &sz) {
        auto p = variables_.find(name);
        // If it exists, update parameter
        if (p != variables_.end()) {
            // Create new symbolic expression and data vector
            variables_[name] = casadi::SX::sym(name, sz);
            variable_map_[name] = Eigen::VectorXd::Zero(sz);
        } else {
            // ! Throw error that parameter is not included
            std::cout << "Variable " << name
                      << "is not in the parameter map!\n";
        }
    }

    /**
     * @brief Sets the optimisation vector for the program to a vector with
     * ordering of variables provided by order
     *
     * @param variables
     */
    void ConstructDecisionVariableVector(
        const std::vector<std::string> &order) {
        // Vector of inputs
        casadi::SXVector x;

        int idx = 0;
        for (int i = 0; i < order.size(); ++i) {
            x.push_back(GetVariables(order[i]));
            variable_idx_[order[i]] = idx;
            idx += GetVariables(order[i]).size1();
        }

        // Create optimisation vector
        x_ = casadi::SX::vertcat(x);

        // Set number of decision variables
        nx_ = x_.size1();

        // Create bounds for the variables
        lbx_.resize(nx_);
        ubx_.resize(nx_);

        lbx_.setConstant(-std::numeric_limits<double>::infinity());
        ubx_.setConstant(std::numeric_limits<double>::infinity());
    }

    /**
     * @brief Sets the optimisation vector for the program to the user-provided
     * vector x
     *
     * @param x
     */
    void ConstructDecisionVariableVector(const casadi::SX &x) { x_ = x; }

    /**
     * @brief Prints the current set of parameters for the program to the
     * screen
     *
     */
    void ListParameters();

    /**
     * @brief Prints the current set of parameters for the program to the
     * screen
     *
     */
    void ListVariables();

    /**
     * @brief Prints the current set of constraints for the program to the
     * screen
     *
     */
    void ListConstraints();

    /**
     * @brief Prints the current set of costs for the program to the
     * screen
     *
     */
    void ListCosts();

    /**
     * @brief Prints a summary of the program, listing number of variables,
     * parameters, constraints and costs
     *
     */
    void PrintProgramSummary();

    /**
     * @brief Name of the program
     *
     * @return const std::string&
     */
    const std::string &name() const { return name_; }

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
        damotion::common::Profiler("Program::AddParameters");
        // If doesn't exist, add parameter
        if (!IsParameter(name)) {
            parameters_[name] = casadi::SX::sym(name, sz);
            parameter_map_[name] = Eigen::VectorXd::Zero(sz);
        } else {
            std::cout << "Parameter " << name
                      << "is not in the parameter map!\n";
        }
    }

    casadi::SX &DecisionVariableVector() { return x_; }

    const int &NumberOfDecisionVariables() const { return nx_; }
    const int &NumberOfConstraints() const { return nc_; }

    std::unordered_map<std::string, Constraint> &Constraints() {
        return constraints_;
    }
    std::unordered_map<std::string, Cost> &Costs() { return costs_; }

    Eigen::VectorXd &DecisionVariablesLowerBound() { return lbx_; }
    Eigen::VectorXd &DecisionVariablesUpperBound() { return ubx_; }

    Eigen::VectorXd &ConstraintsLowerBound() { return lbg_; }
    Eigen::VectorXd &ConstraintsUpperBound() { return ubg_; }

    /**
     * @brief Given the list of input names, assembles a vector of symbolic
     * inputs using the variables and parameters present in the program
     *
     * @param inames
     * @return casadi::SXVector
     */
    casadi::SXVector GetSymbolicFunctionInput(
        const std::vector<std::string> &inames) {
        casadi::SXVector in = {};
        // Gather symbols for each input
        for (int i = 0; i < inames.size(); ++i) {
            std::string name = inames[i];
            if (IsVariable(name)) {
                in.push_back(GetVariables(name));
            } else if (IsParameter(name)) {
                in.push_back(GetParameters(name));
            } else {
                std::cout << "Input " << name
                          << " is not a listed variable or parameter!\n";
                // Return empty vector
                return {};
            }
        }

        // Return inputs
        return in;
    }

   private:
    std::string name_;

    // Number of decision variables
    int nx_ = 0;
    // Number of constraints
    int nc_ = 0;

    Eigen::VectorXd lbx_;
    Eigen::VectorXd ubx_;

    Eigen::VectorXd lbg_;
    Eigen::VectorXd ubg_;

    // Optimisation vector
    casadi::SX x_;

    // Program data

    // Variables
    std::unordered_map<std::string, casadi::SX> variables_;
    std::unordered_map<std::string, casadi::SX> parameters_;

    // Variable indices
    std::unordered_map<std::string, int> variable_idx_;

    // Constraints
    std::unordered_map<std::string, Constraint> constraints_;
    // Costs
    std::unordered_map<std::string, Cost> costs_;

    std::unordered_map<std::string, Eigen::VectorXd> variable_map_;
    // Parameters
    std::unordered_map<std::string, Eigen::VectorXd> parameter_map_;

    // Utilities
    casadi::StringVector GetSXVectorNames(const casadi::SXVector &x);
};

}  // namespace solvers
}  // namespace damotion

#endif /* SOLVERS_PROGRAM_H */
