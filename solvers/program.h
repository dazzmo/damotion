#ifndef SOLVERS_PROGRAM_H
#define SOLVERS_PROGRAM_H

#include <casadi/casadi.hpp>

#include "common/profiler.h"
#include "utils/casadi.h"
#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"

using namespace casadi_utils;

namespace damotion {
namespace solvers {

class Program {
   public:
    Program() = default;
    ~Program() = default;

    Program(const std::string &name) : name_(name) {}

    class Cost {
       public:
        Cost() = default;
        ~Cost() = default;

        /**
         * @brief Create a new Cost object based on the expression f and
         * optimisation variables x
         *
         * @param name Name of the cost
         * @param f Expression for the cost
         * @param in Input variables
         * @param inames Input variable names
         * @param x Optimisation variables
         */
        Cost(const std::string &name, casadi::SX &f, const casadi::SXVector &in,
             const casadi::StringVector &inames, const casadi::SX &x);

        /**
         * @brief Objective function
         *
         */
        eigen::FunctionWrapper obj;

        /**
         * @brief Gradient function
         *
         */
        eigen::FunctionWrapper grad;

        /**
         * @brief Hessian function
         *
         */
        eigen::FunctionWrapper hes;

        /**
         * @brief Input variables for the expression
         *
         */
        std::vector<casadi::SX> in;

        /**
         * @brief Input variable names
         *
         */
        std::vector<std::string> inames;

        /**
         * @brief Output names
         *
         */
        std::vector<std::string> onames;

        /**
         * @brief Cost weighting
         *
         * @return const double
         */
        const double weighting() const { return w_; }

        /**
         * @brief Cost weighting
         *
         * @return double&
         */
        double &weighting() { return w_; }

       private:
        // Cost weighting
        double w_;
    };

    class Constraint {
       public:
        Constraint() = default;
        ~Constraint() = default;

        /**
         * @brief Construct a new Constraint object based on expression f and
         * optimisation variables x
         *
         * @param name Constraint name
         * @param c Constraint
         * @param in Constraint inputs
         * @param inames Constraint input names
         * @param x Decision variable to compute derivatives with respect to
         */
        Constraint(const std::string &name, casadi::SX &f,
                   const casadi::SXVector &in,
                   const casadi::StringVector &inames, const casadi::SX &x);

        /**
         * @brief Name of the constraint
         *
         * @return const std::string&
         */
        const std::string &name() const { return name_; }

        /**
         * @brief Dimension of the constraint
         *
         * @return const int
         */
        const int dim() const { return dim_; }

        /**
         * @brief The index of the constraint within the constraint vector
         *
         * @return const int&
         */
        const int &idx() const { return idx_; }

        /**
         * @brief Set the index of this constraint within the program constraint
         * vector
         *
         * @param idx
         */
        void SetIndex(const int idx) { idx_ = idx; }

        /**
         * @brief Symbolic input variables for the constraint
         *
         */
        casadi::SXVector in;

        // Constraint
        eigen::FunctionWrapper con;
        // Jacobian
        eigen::FunctionWrapper jac;

        /**
         * @brief Constraint lower bound (dim x 1)
         *
         * @return const Eigen::VectorXd&
         */
        const Eigen::VectorXd &lb() const { return lb_; }
        Eigen::VectorXd &lb() { return lb_; }

        /**
         * @brief Constraint upper bound (dim x 1)
         *
         * @return const Eigen::VectorXd&
         */
        const Eigen::VectorXd &ub() const { return ub_; }
        Eigen::VectorXd &ub() { return ub_; }

        // Input names
        std::vector<std::string> inames;
        // Output names
        std::vector<std::string> onames;

        /**
         * @brief Tests whether the p-norm of the constraint is within
         * the threshold eps.
         *
         * @param p The norm of the constraint (use Eigen::Infinity for the
         * infinity norm)
         * @param eps
         * @return true
         * @return false
         */
        bool CheckViolation(const int &p = 2, const double &eps = 1e-6) {
            // Determine if constraint within threshold
            double c_norm = 0.0;
            if (p == 1) {
                c_norm = con.getOutput(0).lpNorm<1>();
            } else if (p == 2) {
                c_norm = con.getOutput(0).lpNorm<2>();
            } else if (p == Eigen::Infinity) {
                c_norm = con.getOutput(0).lpNorm<Eigen::Infinity>();
            }

            return c_norm <= eps;
        }

       private:
        int dim_ = 0;

        int idx_ = 0;

        std::string name_;

        // Constraint lower bound
        Eigen::VectorXd lb_;
        // Constraint upper bound
        Eigen::VectorXd ub_;
    };

    void AddVariables(const std::string &name, const int sz) {
        if (variables_.find(name) == variables_.end()) {
            variables_[name] = casadi::SX::sym(name, sz);
            variable_map_[name] = Eigen::VectorXd::Zero(sz);
        } else {
            // ! Make this more sophisticated
            std::cout << "Variables with name " << name << " already added!\n";
        }
    }

    casadi::SX &GetVariables(const std::string &name) {
        if (variables_.find(name) == variables_.end()) {
            // return casadi::SX(0);
        } else {
            return variables_[name];
        }
    }

    casadi::SX &GetParameters(const std::string &name) {
        if (parameters_.find(name) == parameters_.end()) {
            // return casadi::SX(0);
        } else {
            return parameters_[name];
        }
    }

    int GetVariableIndex(const std::string &name) {
        if (variable_idx_.find(name) == variable_idx_.end()) {
            std::cout << "Variable " << name
                      << " is not within this program!\n";
            return -1;
        } else {
            return variable_idx_[name];
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

    void AddCost(const std::string &name, Cost &cost) {
        // Register constraint inputs with variables and parameters
        SetFunctionData(cost.obj);
        SetFunctionData(cost.grad);
        SetFunctionData(cost.hes);

        // Add to cost map
        costs_[name] = cost;
    }

    Cost &GetCost(std::string &name) { return costs_[name]; }

    void AddConstraint(const std::string &name, Constraint &con) {
        // Register constraint inputs with data
        SetFunctionData(con.con);
        SetFunctionData(con.jac);

        // Add to constraint map
        constraints_[name] = con;
    }

    Constraint &GetConstraint(std::string &name) { return constraints_[name]; }

    void SetFunctionData(eigen::FunctionWrapper &f) {
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
    void ListParameters() {
        std::cout << "Program " << name() << '\n';
        std::cout << "Parameter\tSize\n";
        for (auto p : parameter_map_) {
            std::cout << p.first << "\t" << p.second.size() << " x 1\n";
        }
    }

    /**
     * @brief Prints the current set of parameters for the program to the
     * screen
     *
     */
    void ListVariables() {
        std::cout << "Program " << name() << '\n';
        std::cout << "Variable\tSize\n";
        for (auto v : variables_) {
            std::cout << v.first << "\t" << v.second.size() << " x 1\n";
        }
    }

    /**
     * @brief Prints the current set of constriants for the program to the
     * screen
     *
     */
    void ListConstraints() {
        std::cout << "Program " << name() << '\n';
        std::cout << "Constraint\tSize\n";
        for (auto c : constraints_) {
            std::cout << c.first << "\t" << c.second.dim() << " x 1\n";
        }
    }

    /**
     * @brief Prints the current set of costs for the program to the
     * screen
     *
     */
    void ListCosts() {
        std::cout << "Program " << name() << '\n';
        std::cout << "Cost\tWeighting\n";
        for (auto c : costs_) {
            std::cout << c.first << "\t" << c.second.weighting() << '\n';
        }
    }

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
        // Look up parameters in parameter map and set values
        auto p = parameters_.find(name);
        // If doesn't exist, add parameter
        if (p == parameters_.end()) {
            parameters_[name] = casadi::SX::sym(name, sz);
            parameter_map_[name] = Eigen::VectorXd::Zero(sz);
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
};

}  // namespace solvers
}  // namespace damotion

#endif /* SOLVERS_PROGRAM_H */
