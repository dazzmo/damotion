#ifndef SOLVERS_PROGRAM_H
#define SOLVERS_PROGRAM_H

#include <casadi/casadi.hpp>

#include "common/profiler.h"
#include "solvers/binding.h"
#include "solvers/constraint.h"
#include "solvers/cost.h"
#include "symbolic/expression.h"
#include "symbolic/variable.h"
#include "utils/casadi.h"
#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"

namespace sym = damotion::symbolic;

namespace damotion {
namespace optimisation {

// Forward declaration of SolverBase
namespace solvers {
class SolverBase;
}

/**
 * @brief The program class creates mathematical programs.
 * It uses a block structure to allow greater speed in accessing data, thus
 * optimisation variables are grouped in sets, which can be as small as a single
 * unit.
 *
 */
class Program {
   public:
    Program() = default;
    ~Program() = default;

    friend class solvers::SolverBase;

    Program(const std::string &name)
        : name_(name),
          n_decision_variables_(0),
          n_constraints_(0),
          n_parameters_(0) {}

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
    const int &NumberOfDecisionVariables() const {
        return n_decision_variables_;
    }

    /**
     * @brief Number of constraints currently in the program
     *
     * @return const int&
     */
    const int &NumberOfConstraints() const { return n_constraints_; }

    /**
     * @brief Number of parameters currently in the program
     *
     * @return const int&
     */
    const int &NumberOfParameters() const { return n_parameters_; }

    /**
     * @brief Add decision variables to the program
     *
     * @param var
     */
    void AddDecisionVariables(const Eigen::Ref<sym::VariableMatrix> &var);

    /**
     * @brief Adds a single decision variable to the program
     *
     * @param var
     */
    void AddDecisionVariable(const sym::Variable &var);

    /**
     * @brief Removes variables currently considered by the program.
     *
     * @param var
     */
    void RemoveDecisionVariables(const Eigen::Ref<sym::VariableMatrix> &var);

    int GetDecisionVariableIndex(const sym::Variable &v);

    bool IsDecisionVariable(const sym::Variable &var);

    /**
     * @brief Add parameters to the program
     *
     * @param var
     */
    Eigen::Ref<const Eigen::MatrixXd> AddParameters(const std::string &name,
                                                    int n, int m = 1);

    Eigen::Ref<const Eigen::MatrixXd> GetParameters(const std::string &name);

    void SetParameters(const std::string &name,
                       Eigen::Ref<const Eigen::MatrixXd> val);

    /**
     * @brief Removes variables currently considered by the program.
     *
     * @param var
     */
    void RemoveParameters(const std::string &name);

    Binding<Cost> AddCost(const std::shared_ptr<Cost> &cost,
                          const sym::VariableRefVector &x,
                          const sym::ParameterRefVector &p);

    // TODO - Remove cost or constraint

    Binding<LinearConstraint> AddLinearConstraint(
        const std::shared_ptr<LinearConstraint> &con,
        const sym::VariableRefVector &x, const sym::ParameterRefVector &p);

    Binding<BoundingBoxConstraint> AddBoundingBoxConstraint(
        const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
        const sym::VariableVector &x);

    Binding<BoundingBoxConstraint> AddBoundingBoxConstraint(
        const double &lb, const double &ub, const sym::VariableVector &x);

    /**
     * @brief Add a constraint to the program that uses the variables
     * and parameters given by x and p respectively.
     *
     * @param c
     * @param x
     * @param p
     * @return Binding<Constraint>
     */
    Binding<Constraint> AddConstraint(const std::shared_ptr<Constraint> &c,
                                      const sym::VariableRefVector &x,
                                      const sym::ParameterRefVector &p);

    /**
     * @brief Add a generic constraint to the program that uses the variables
     * and parameters given by x and p respectively.
     *
     * @param c
     * @param x
     * @param p
     * @return Binding<Constraint>
     */
    Binding<Constraint> AddGenericConstraint(std::shared_ptr<Constraint> &c,
                                             const sym::VariableRefVector &x,
                                             const sym::ParameterRefVector &p);

    std::vector<Binding<Constraint>> GetAllConstraints() {
        std::vector<Binding<Constraint>> constraints;
        constraints.insert(constraints.begin(), linear_constraints_.begin(),
                           linear_constraints_.end());

        // Return vector of all constraints
        return constraints;
    }

    /**
     * @brief Decision variable vector for the program.
     *
     * @return const Eigen::VectorXd &
     */
    const Eigen::VectorXd &DecisionVariableVector() const { return x_; }

    Eigen::VectorXd &DecisionVariablesLowerBound() { return lbx_; }
    Eigen::VectorXd &DecisionVariablesUpperBound() { return ubx_; }

    Eigen::VectorXd &ConstraintsLowerBound() { return lbg_; }
    Eigen::VectorXd &ConstraintsUpperBound() { return ubg_; }

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
     * @brief Sets the optimisation vector with the given ordering of variables
     *
     * @param order
     */
    bool SetDecisionVariableVector(const Eigen::Ref<sym::VariableVector> &var);

    /**
     * @brief Set the optimisation to the default ordering of variables (ordered
     * by when they were added to the program)
     *
     */
    void SetDecisionVariableVector();

    /**
     * @brief Get the vector of current LinearConstraint objects within the
     * program.
     *
     * @return std::vector<Binding<LinearConstraint>>&
     */
    std::vector<Binding<LinearConstraint>> &GetLinearConstraintBindings() {
        // Create constraints
        return linear_constraints_;
    }

    /**
     * @brief Get the vector of current BoundingBoxConstraint objects within the
     * program.
     *
     * @return std::vector<Binding<BoundingBoxConstraint>>&
     */
    std::vector<Binding<BoundingBoxConstraint>> &
    GetBoundingBoxConstraintBindings() {
        // Create constraints
        return bounding_box_constraints_;
    }

    std::vector<Binding<Cost>> &GetCostBindings() {
        // TODO - Look at different classes of costs
        return costs_;
    }

    // Update all bindings and indices
    void UpdateBindings() {
        // Go through all types of constraints to update

        // Add Sparse Jacobian stuff here as well
    }

   private:
    // Program name
    std::string name_;

    // Number of decision variables
    int n_decision_variables_ = 0;
    // Number of constraints
    int n_constraints_ = 0;
    // Number of parameters
    int n_parameters_ = 0;

    // Decision variables lower bound
    Eigen::VectorXd lbx_;
    // Decision variables upper bound
    Eigen::VectorXd ubx_;

    // Constrain vector lower bound
    Eigen::VectorXd lbg_;
    // Constrain vector upper bound
    Eigen::VectorXd ubg_;

    // Optimisation vector
    Eigen::VectorXd x_;
    // Parameter vector
    typedef sym::Variable Parameter;
    Parameter p_;

    // Decision variables
    std::unordered_map<sym::Variable::Id, int> decision_variable_idx_;
    std::vector<sym::Variable> decision_variables_;

    // Parameters
    std::unordered_map<std::string, Eigen::MatrixXd> parameters_;

    // Constraints
    std::vector<Binding<Constraint>> constraints_;
    std::vector<Binding<LinearConstraint>> linear_constraints_;
    std::vector<Binding<BoundingBoxConstraint>> bounding_box_constraints_;

    // Costs
    std::vector<Binding<Cost>> costs_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_PROGRAM_H */
