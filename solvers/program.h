#ifndef SOLVERS_PROGRAM_H
#define SOLVERS_PROGRAM_H

#include <casadi/casadi.hpp>

#include "common/profiler.h"
#include "solvers/constraint.h"
#include "solvers/cost.h"
#include "solvers/variable.h"
#include "utils/casadi.h"
#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"

namespace damotion {
namespace optimisation {

// Forward declaration of SolverBase
namespace solvers {
class SolverBase;
}

class Program {
   public:
    Program() = default;
    ~Program() = default;

    friend class solvers::SolverBase;

    Program(const std::string &name) : name_(name), nx_(0), nc_(0), np_(0) {}

    /**
     * @brief Name of the program
     *
     * @return const std::string&
     */
    const std::string &name() const { return name_; }

    /**
     * @brief Number of decision variables currently in the program
     *
     * @return const int&
     */
    const int &NumberOfDecisionVariables() const { return nx_; }

    /**
     * @brief Number of constraints currently in the program
     *
     * @return const int&
     */
    const int &NumberOfConstraints() const { return nc_; }

    /**
     * @brief Number of parameters currently in the program
     *
     * @return const int&
     */
    const int &NumberOfParameters() const { return np_; }

    /**
     * @brief Adds symbolic variable vector to the list of variables within the
     * program. Duplicate names are not allowed.
     *
     * @param name
     * @param n Number of rows
     * @param m Number of columns
     */
    void AddDecisionVariables(const std::string &name, const int n,
                              const int m = 1);

    /**
     * @brief Removes a variable currently considered by the program.
     *
     * @param name
     */
    void RemoveDecisionVariables(const std::string &name);

    Variable &GetDecisionVariables(const std::string &name);

    /**
     * @brief The index of the variables given by name within the decision
     * variable vector given by DecisionVariableVector()
     *
     * @param name
     * @return int
     */
    int GetDecisionVariablesIndex(const std::string &name);

    /**
     * @brief Resizes a decision variable to a new size sz.
     *
     * @param name Name of the decision variables
     * @param sz The new size
     */
    void ResizeDecisionVariables(const std::string &name, const int n,
                                 const int m = 1);

    /**
     * @brief Indicates whether the provided variable is present within the
     * problem
     *
     * @param name
     * @return true
     * @return false
     */
    bool IsDecisionVariable(const std::string &name);

    void SetParameters(const std::string &name, const Eigen::MatrixXd &val);
    void AddParameters(const std::string &name, const int n, const int m = 1);
    void RemoveParameters(const std::string &name);
    Variable &GetParameters(const std::string &name);

    /**
     * @brief Indicates whether the provided parameter is present within the
     * problem
     *
     * @param name
     * @return true
     * @return false
     */
    inline bool IsParameter(const std::string &name);

    /**
     * @brief Adds a cost to the program given by the expression cost and inputs
     * in. Cost gradients can be computed with respect to a custom list of
     * variables given by x and xnames. Similarly, cost hessians can also be
     * computed by providing custom variable pairings in xy and xynames.
     *
     * @param name Name of the cost
     * @param cost Expression for cost
     * @param in Inputs used to generate cost
     * @param x Variables to compute gradient of cost with respect to
     * @param xnames Names of variables in x
     * @param xy Variable pairs to compute cost hessian with respect to
     * @param xynames Names of variables in xy
     */
    void AddCost(
        const std::string &name, casadi::SX &cost, casadi::SXVector &in,
        const casadi::SXVector &x = {}, const casadi::StringVector xnames = {},
        const std::vector<std::pair<::casadi::SX, ::casadi::SX>> &xy = {},
        const std::vector<std::pair<std::string, std::string>> &xynames = {});

    void RemoveCost(const std::string &name);
    Cost &GetCost(const std::string &name);

    /**
     * @brief Adds a constraint to the program given by the expression
     * constraint and inputs in. Jacobians can be computed with respect to a
     * custom list of variables given by x and xnames, otherwise the jaocbian is
     * only computed for the current decision variable vector given by
     * DecisionVariableVector(). Similarly, hessians can also be computed
     * by providing custom variable pairings in xy and xynames.
     *
     * @param name
     * @param constraint
     * @param in
     * @param bounds
     * @param x
     * @param xnames
     * @param xy
     * @param xynames
     */
    void AddConstraint(
        const std::string &name, casadi::SX &constraint, casadi::SXVector &in,
        const BoundsType &bounds = BoundsType::kUnbounded,
        const casadi::SXVector &x = {}, const casadi::StringVector xnames = {},
        const std::vector<std::pair<::casadi::SX, ::casadi::SX>> &xy = {},
        const std::vector<std::pair<std::string, std::string>> &xynames = {});

    Constraint &GetConstraint(const std::string &name);
    void RemoveConstraint(const std::string &name);

    /**
     * @brief The index of the variables given by name within the decision
     * variable vector given by DecisionVariableVector()
     *
     * @param name
     * @return int
     */
    int GetConstraintIndex(const std::string &name);

    /**
     * @brief Decision variable vector for the program.
     *
     * @return Variable&
     */
    Variable &DecisionVariableVector() { return x_; }

    Eigen::VectorXd &DecisionVariablesLowerBound() { return lbx_; }
    Eigen::VectorXd &DecisionVariablesUpperBound() { return ubx_; }

    Eigen::VectorXd &ConstraintsLowerBound() { return lbg_; }
    Eigen::VectorXd &ConstraintsUpperBound() { return ubg_; }

    /**
     * @brief Updates the bounds for the decision variable vector for the
     * variable given by name
     *
     * @param name
     */
    void UpdateDecisionVariableVectorBounds(const std::string &name);

    /**
     * @brief Updates the bounds for the decision variable vector for all
     * decision variables
     *
     */
    void UpdateDecisionVariableVectorBounds();

    void UpdateConstraintVectorBounds(const std::string &name);
    void UpdateConstraintVectorBounds();

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
     * @brief Returns the id of a cost within the costs vector of the program.
     * If the cost does not exist, returns -1.
     *
     * @param name
     * @return  int
     */
    int GetCostId(const std::string &name);

    /**
     * @brief Returns the id of a constraint within the constraints vector of
     * the program. If the constraint does not exist, returns -1.
     *
     * @param name
     * @return  int
     */
    int GetConstraintId(const std::string &name);

   protected:
    /**
     * @brief Returns the id of a variable within the variables vectors of the
     * program. If the variable does not exist, returns -1.
     *
     * @param name
     * @return  int
     */
    int GetDecisionVariablesId(const std::string &name);

    /**
     * @brief Returns the id of a parameter within the parameters vectors of
     * the program. If the parameter does not exist, returns -1.
     *
     * @param name
     * @return  int
     */
    int GetParametersId(const std::string &name);

    void ConstructConstraintVector();

    /**
     * @brief Given the list of input names, assembles a vector of symbolic
     * inputs using the variables and parameters present in the program
     *
     * @param inames
     * @return casadi::SXVector
     */
    casadi::SXVector GetSymbolicFunctionInput(
        const std::vector<std::string> &inames);

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

    // void FormatFunctionInput();

    /**
     * @brief Get the vector of current Cost objects within the program
     *
     * @return std::vector<Cost>&
     */
    std::vector<Cost> &GetCosts() { return costs_; }

    /**
     * @brief Get the vector of current Constraint objects within the program.
     *
     * @return std::vector<Constraint>&
     */
    std::vector<Constraint> &GetConstraints() { return constraints_; }

    /**
     * @brief Sets the optimisation vector to the values given in the vector x
     *
     * @param x
     */
    void SetDecisionVariableVector(const Eigen::VectorXd &x) { x_.val() = x; }

    /**
     * @brief Sets each decision variable within GetVariables() to the values
     * within the vector x
     *
     * @param x
     */
    void SetDecisionVariablesFromVector(const Eigen::VectorXd &x) {
        // Use current mapping to compute variables
        for (auto &xi : variables_id_) {
            int id = xi.second;
            Variable &v = variables_[id];
            v.val() << x.middleRows(v.idx().i_start(), v.idx().i_sz());
        }
    }

   private:
    // Program name
    std::string name_;

    // Number of decision variables
    int nx_ = 0;
    // Number of constraints
    int nc_ = 0;
    // Number of parameters
    int np_ = 0;

    // Decision variables lower bound
    Eigen::VectorXd lbx_;
    // Decision variables upper bound
    Eigen::VectorXd ubx_;

    // Constrain vector lower bound
    Eigen::VectorXd lbg_;
    // Constrain vector upper bound
    Eigen::VectorXd ubg_;

    // Optimisation vector
    Variable x_;

    // Variables
    std::unordered_map<std::string, int> variables_id_;
    std::vector<BlockIndex> variables_idx_;
    std::vector<Variable> variables_;

    // Parameters
    std::unordered_map<std::string, int> parameters_id_;
    std::vector<Variable> parameters_;

    // Constraints
    std::unordered_map<std::string, int> constraints_id_;
    std::vector<BlockIndex> constraints_idx_;
    std::vector<std::vector<BlockIndex>> constraints_jac_idx_;
    std::vector<std::vector<BlockIndex>> constraints_hes_idx_;
    std::vector<Constraint> constraints_;

    // Costs
    std::unordered_map<std::string, int> costs_id_;
    std::vector<std::vector<BlockIndex>> costs_grad_idx_;
    std::vector<std::vector<BlockIndex>> costs_hes_idx_;

    std::vector<Cost> costs_;

    // Utilities
    std::string GetSXName(const casadi::SX &x);
    casadi::StringVector GetSXVectorNames(const casadi::SXVector &x);
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_PROGRAM_H */
