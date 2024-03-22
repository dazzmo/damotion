#ifndef SOLVERS_PROGRAM_H
#define SOLVERS_PROGRAM_H

#include <casadi/casadi.hpp>

#include "common/profiler.h"
#include "solvers/constraint.h"
#include "solvers/cost.h"
#include "utils/casadi.h"
#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"

namespace damotion {
namespace optimisation {

class Program {
   public:
    Program() = default;
    ~Program() = default;

    Program(const std::string &name) : name_(name) {}

    /**
     * @brief Name of the program
     *
     * @return const std::string&
     */
    const std::string &name() const { return name_; }

    /**
     * @brief Adds symbolic variable vector to the list of variables within the
     * program. Duplicate names are not allowed.
     *
     * @param name
     * @param sz
     */
    void AddVariables(const std::string &name, const int sz);

    /**
     * @brief Returns the symbolic vector for the variable given by name
     *
     * @param name
     * @return casadi::SX&
     */
    casadi::SX &GetVariables(const std::string &name);

    /**
     * @brief The starting index of the variables given by name within the
     * DecisionVaraibleVector() x
     *
     * @param name
     * @return int
     */
    int GetVariableIndex(const std::string &name);

    /**
     * @brief Returns the symbolic vector for the parameter given by name
     *
     * @param name
     * @return casadi::SX&
     */
    casadi::SX &GetParameters(const std::string &name);

    Eigen::Ref<Eigen::VectorXd> GetVariableUpperBounds(const std::string &name);

    Eigen::Ref<Eigen::VectorXd> GetVariableLowerBounds(const std::string &name);

    /**
     * @brief Indicates whether the provided variable is present within the
     * problem
     *
     * @param name
     * @return true
     * @return false
     */
    inline bool IsVariable(const std::string &name);

    /**
     * @brief Given the decision variable vector x, set all variables within the
     * program using the values from x.
     *
     * @param x
     */
    void SetDecisionVariablesFromVector(const Eigen::VectorXd &x);

    /**
     * @brief Resizes a decision variable to a new size sz.
     *
     * @param name Name of the decision variables
     * @param sz The new size
     */
    void ResizeDecisionVariables(const std::string &name, const int &sz);

    /**
     * @brief Indicates whether the provided parameter is present within the
     * problem
     *
     * @param name
     * @return true
     * @return false
     */
    inline bool IsParameter(const std::string &name);

    void SetParameter(const std::string &name, const Eigen::VectorXd &val);
    void AddParameters(const std::string &name, int sz);

    void AddCost(const std::string &name, casadi::SX &cost,
                 casadi::SXVector &in);

    void AddConstraint(const std::string &name, casadi::SX &constraint,
                       casadi::SXVector &in, const BoundsType &bounds = BoundsType::kUnbounded);

    Cost &GetCost(const std::string &name) { return costs_[name]; }

    Constraint &GetConstraint(const std::string &name) {
        return constraints_[name];
    }

    void SetUpCost(Cost &cost);
    void SetUpCosts();

    void SetUpConstraint(Constraint &constraint);
    void SetUpConstraints();

    void ConstructConstraintVector();

    /**
     * @brief For a given FunctionWrapper f, sets the inputs of the function to
     * the variable and parameter data vectors within the program so future
     * calls to f will use the most recent values of the variables and
     * parameters.
     *
     * @param f
     */
    void SetFunctionInputData(utils::casadi::FunctionWrapper &f);

    /**
     * @brief Sets the optimisation vector for the program to a vector with
     * ordering of variables provided by order
     *
     * @param order
     */
    void ConstructDecisionVariableVector(const std::vector<std::string> &order);

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

    casadi::SX &DecisionVariableVector() { return x_; }

    const int &NumberOfDecisionVariables() const { return nx_; }
    const int &NumberOfConstraints() const { return nc_; }

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
        const std::vector<std::string> &inames);

   private:
    // Program name
    std::string name_;

    // Number of decision variables
    int nx_ = 0;
    // Number of constraints
    int nc_ = 0;

    // Decision variables lower bound
    Eigen::VectorXd lbx_;
    // Decision variables upper bound
    Eigen::VectorXd ubx_;

    // Constrain vector lower bound
    Eigen::VectorXd lbg_;
    // Constrain vector upper bound
    Eigen::VectorXd ubg_;

    // Symbolic decision variables vector
    casadi::SX x_;

    // Symbolic variables map
    std::unordered_map<std::string, casadi::SX> variables_;
    // Symbolic parameters map
    std::unordered_map<std::string, casadi::SX> parameters_;

    // Variable indices
    std::unordered_map<std::string, int> variable_idx_;

    // Constraints
    std::unordered_map<std::string, Constraint> constraints_;
    // Costs
    std::unordered_map<std::string, Cost> costs_;

    // Variable data map
    std::unordered_map<std::string, Eigen::VectorXd> variable_map_;
    // Parameter data map
    std::unordered_map<std::string, Eigen::VectorXd> parameter_map_;

    // Utilities
    casadi::StringVector GetSXVectorNames(const casadi::SXVector &x);

    void VariableNotFoundError(const std::string &name);
    void ParameterNotFoundError(const std::string &name);

    void VariableAlreadyExistsError(const std::string &name);
    void ParameterAlreadyExistsError(const std::string &name);
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_PROGRAM_H */
