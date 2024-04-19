#ifndef SOLVERS_PROGRAM_H
#define SOLVERS_PROGRAM_H

#include <casadi/casadi.hpp>

#include "common/profiler.h"
#include "solvers/binding.h"
#include "solvers/constraint.h"
#include "solvers/cost.h"
#include "symbolic/expression.h"
#include "symbolic/parameter.h"
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
 * @brief Class that maintains and adjusts decision variables organised into a
 * vector
 *
 */
class DecisionVariableManager {
   public:
    DecisionVariableManager() : n_decision_variables_(0) {}
    ~DecisionVariableManager() = default;

    /**
     * @brief Number of decision variables currently in the program
     *
     * @return const int&
     */
    const int &NumberOfDecisionVariables() const {
        return n_decision_variables_;
    }

    /**
     * @brief Adds a decision variable
     *
     * @param var
     */
    void AddDecisionVariable(const sym::Variable &var);

    /**
     * @brief Add decision variables
     *
     * @param var
     */
    void AddDecisionVariables(const Eigen::Ref<sym::VariableMatrix> &var);

    /**
     * @brief Removes variables currently considered by the program.
     *
     * @param var
     */
    void RemoveDecisionVariables(const Eigen::Ref<sym::VariableMatrix> &var);

    /**
     * @brief Whether a variable var is a decision variable within the program
     *
     * @param var
     * @return true
     * @return false
     */
    bool IsDecisionVariable(const sym::Variable &var);

    /**
     * @brief Returns the index of the given variable within the created
     * optimisation vector
     *
     * @param v
     * @return int
     */
    int GetDecisionVariableIndex(const sym::Variable &v);

    /**
     * @brief Set the vector of decision variables to the default ordering of
     * variables (ordered by when they were added)
     *
     */
    void SetDecisionVariableVector();

    /**
     * @brief Sets the optimisation vector with the given ordering of variables
     *
     * @param var
     */
    bool SetDecisionVariableVector(const Eigen::Ref<sym::VariableVector> &var);

    /**
     * @brief Prints the current set of parameters for the program to the
     * screen
     *
     */
    void ListDecisionVariables();

   private:
    // Number of decision variables
    int n_decision_variables_;

    std::unordered_map<sym::Variable::Id, int> decision_variable_idx_;
    std::vector<sym::Variable> decision_variables_;
};

/**
 * @brief Class that maintains and adjusts parameters
 *
 */
class ParameterManager {
   public:
    ParameterManager() : n_parameters_(0) {}
    ~ParameterManager() = default;

    /**
     * @brief Number of parameters currently in the program
     *
     * @return const int&
     */
    const int &NumberOfParameters() const { return n_parameters_; }

    /**
     * @brief Whether the parameter par is included within the program
     *
     * @param par
     * @return true
     * @return false
     */
    bool IsParameter(const sym::Parameter &par);

    /**
     * @brief Add parameters to the program
     *
     * @param var
     */
    Eigen::Ref<Eigen::MatrixXd> AddParameter(const sym::Parameter &p);

    /**
     * @brief Returns the values of the parameter p as an Eigen::MatrixXd
     * reference
     *
     * @param p
     * @return Eigen::Ref<const Eigen::MatrixXd>
     */
    Eigen::Ref<const Eigen::MatrixXd> GetParameterValues(
        const sym::Parameter &p);

    /**
     * @brief Sets the parameter p within the program to the values given by
     * val.
     *
     * @param p
     * @param val
     */
    void SetParameterValues(const sym::Parameter &p,
                            Eigen::Ref<const Eigen::MatrixXd> val);

    /**
     * @brief Removes variables currently considered by the program.
     *
     * @param var
     */
    void RemoveParameters(const sym::Parameter &p);

    /**
     * @brief Prints the current set of parameters for the program to the
     * screen
     *
     */
    void ListParameters();

   private:
    // Number of parameters
    int n_parameters_;

    // Parameters
    std::unordered_map<sym::Parameter::Id, int> parameter_idx_;
    std::vector<sym::Parameter> parameters_;
    std::vector<Eigen::MatrixXd> parameter_vals_;
};

template <typename MatrixType = Eigen::MatrixXd>
class CostManager {
   public:
    typedef CostBase<MatrixType> Cost;

    CostManager() = default;
    ~CostManager() = default;

    /**
     * @brief Adds a generic cost
     *
     * @param cost
     * @param x
     * @param p
     * @return Binding<Cost<MatrixType>>
     */
    Binding<Cost> AddCost(const std::shared_ptr<Cost> &cost,
                          const sym::VariableRefVector &x,
                          const sym::ParameterVector &p) {
        Binding<Cost> binding(cost, x, p);
        costs_.push_back(binding);
        return costs_.back();
    }

    Binding<LinearCost<MatrixType>> AddLinearCost(
        const std::shared_ptr<LinearCost<MatrixType>> &cost,
        const sym::VariableRefVector &x, const sym::ParameterVector &p) {
        linear_costs_.push_back(Binding<LinearCost<MatrixType>>(cost, x, p));
        return linear_costs_.back();
    }

    Binding<QuadraticCost<MatrixType>> AddQuadraticCost(
        const std::shared_ptr<QuadraticCost<MatrixType>> &cost,
        const sym::VariableRefVector &x, const sym::ParameterVector &p) {
        quadratic_costs_.push_back(
            Binding<QuadraticCost<MatrixType>>(cost, x, p));
        return quadratic_costs_.back();
    }

    std::vector<Binding<Cost>> GetAllCostBindings() {
        std::vector<Binding<Cost>> costs;
        costs.insert(costs.begin(), linear_costs_.begin(), linear_costs_.end());
        costs.insert(costs.begin(), quadratic_costs_.begin(),
                     quadratic_costs_.end());

        // Return vector of all costs
        return costs;
    }

    /**
     * @brief Prints the current set of costs for the program to the
     * screen
     *
     */
    void ListCosts() {
        std::cout << "----------------------\n";
        std::cout << "Cost\n";
        std::cout << "----------------------\n";
        std::vector<Binding<Cost>> costs = GetAllCostBindings();
        for (Binding<Cost> &b : costs) {
            std::cout << b.Get().name() << '\n';
        }
    }

   private:
    // Costs
    std::vector<Binding<Cost>> costs_;
    std::vector<Binding<LinearCost<MatrixType>>> linear_costs_;
    std::vector<Binding<QuadraticCost<MatrixType>>> quadratic_costs_;
};

template <typename MatrixType = Eigen::MatrixXd>
class ConstraintManager {
   public:
    ConstraintManager() : n_constraints_(0) {}
    ~ConstraintManager() = default;

    /**
     * @brief Number of constraints currently in the program
     *
     * @return const int&
     */
    const int &NumberOfConstraints() const { return n_constraints_; }

    /**
     * @brief Add a constraint to the program that uses the variables
     * and parameters given by x and p respectively.
     *
     * @param c
     * @param x
     * @param p
     * @return Binding<ConstraintBase>
     */
    Binding<Constraint> AddConstraint(const std::shared_ptr<Constraint> &con,
                                      const sym::VariableRefVector &x,
                                      const sym::ParameterVector &p) {
        // Create a binding for the constraint
        constraints_.push_back(Binding<Constraint>(con, x, p));
        n_constraints_ += con->Dimension();
        return constraints_.back();
    }

    Binding<LinearConstraint<MatrixType>> AddLinearConstraint(
        const std::shared_ptr<LinearConstraint<MatrixType>> &con,
        const sym::VariableRefVector &x, const sym::ParameterVector &p) {
        linear_constraints_.push_back(
            Binding<LinearConstraint<MatrixType>>(con, x, p));
        n_constraints_ += con->Dimension();
        return linear_constraints_.back();
    }

    Binding<BoundingBoxConstraint<MatrixType>> AddBoundingBoxConstraint(
        const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
        const sym::VariableVector &x) {
        std::shared_ptr<BoundingBoxConstraint<MatrixType>> con =
            std::make_shared<BoundingBoxConstraint<MatrixType>>("", lb, ub);
        bounding_box_constraints_.push_back(
            Binding<BoundingBoxConstraint<MatrixType>>(con, {x}));
        return bounding_box_constraints_.back();
    }

    Binding<BoundingBoxConstraint<MatrixType>> AddBoundingBoxConstraint(
        const double &lb, const double &ub, const sym::VariableVector &x) {
        Eigen::VectorXd lbv(x.size()), ubv(x.size());
        lbv.setConstant(lb);
        ubv.setConstant(ub);
        return AddBoundingBoxConstraint(lbv, ubv, x);
    }

    std::vector<Binding<ConstraintBase<MatrixType>>>
    GetAllConstraintBindings() {
        std::vector<Binding<ConstraintBase<MatrixType>>> constraints;
        constraints.insert(constraints.begin(), linear_constraints_.begin(),
                           linear_constraints_.end());

        // Return vector of all constraints
        return constraints;
    }

    /**
     * @brief Get the vector of current LinearConstraint objects within the
     * program.
     *
     * @return std::vector<Binding<LinearConstraint>>&
     */
    std::vector<Binding<LinearConstraint<MatrixType>>> &
    GetLinearConstraintBindings() {
        // Create constraints
        return linear_constraints_;
    }

    /**
     * @brief Get the vector of current BoundingBoxConstraint objects within the
     * program.
     *
     * @return std::vector<Binding<BoundingBoxConstraint>>&
     */
    std::vector<Binding<BoundingBoxConstraint<MatrixType>>> &
    GetBoundingBoxConstraintBindings() {
        // Create constraints
        return bounding_box_constraints_;
    }

    void ListConstraints() {
        std::cout << "----------------------\n";
        std::cout << "Constraint\tSize\tLower Bound\tUpper Bound\n";
        std::cout << "----------------------\n";
        // Get all constraints
        std::vector<Binding<ConstraintBase<MatrixType>>> constraints =
            GetAllConstraintBindings();
        for (Binding<ConstraintBase<MatrixType>> &b : constraints) {
            std::cout << b.Get().name() << "\t[" << b.Get().Dimension()
                      << ",1]\n";
            for (int i = 0; i < b.Get().Dimension(); ++i) {
                std::cout << b.Get().name() << "_" + std::to_string(i) << "\t\t"
                          << b.Get().LowerBound()[i] << "\t"
                          << b.Get().UpperBound()[i] << "\n";
            }
        }
        for (Binding<BoundingBoxConstraint<MatrixType>> &b :
             GetBoundingBoxConstraintBindings()) {
            std::cout << b.Get().name() << "\t[" << b.Get().Dimension()
                      << ",1]\n";
            for (int i = 0; i < b.Get().Dimension(); ++i) {
                std::cout << b.Get().name() << "_" + std::to_string(i) << "\t\t"
                          << b.Get().LowerBound()[i] << "\t"
                          << b.Get().UpperBound()[i] << "\n";
            }
        }
    }

   private:
    // Number of constraints
    int n_constraints_;

    std::vector<Binding<ConstraintBase<MatrixType>>> constraints_;
    std::vector<Binding<LinearConstraint<MatrixType>>> linear_constraints_;
    std::vector<Binding<BoundingBoxConstraint<MatrixType>>>
        bounding_box_constraints_;
};

/**
 * @brief The program class creates mathematical programs.
 * It uses a block structure to allow greater speed in accessing data, thus
 * optimisation variables are grouped in sets, which can be as small as a single
 * unit.
 *
 */
template <typename MatrixType>
class ProgramBase : public DecisionVariableManager,
                    public ParameterManager,
                    public CostManager<MatrixType>,
                    public ConstraintManager<MatrixType> {
   public:
    ProgramBase() = default;
    ~ProgramBase() = default;

    friend class solvers::SolverBase;

    ProgramBase(const std::string &name) : name_(name) {}

    /**
     * @brief Name of the program
     *
     * @return const std::string&
     */
    const std::string &name() const { return name_; }

    /**
     * @brief Decision variable vector for the program.
     *
     * @return const Eigen::VectorXd &
     */
    const Eigen::VectorXd &DecisionVariableVector() const { return x_; }

    Eigen::VectorXd &ConstraintsLowerBound() { return lbg_; }
    Eigen::VectorXd &ConstraintsUpperBound() { return ubg_; }

    /**
     * @brief Prints a summary of the program, listing number of variables,
     * parameters, constraints and costs
     *
     */
    void PrintProgramSummary() {
        std::cout << "-----------------------\n";
        std::cout << "Program Name: " << name() << '\n';
        std::cout << "Number of Decision Variables: "
                  << this->NumberOfDecisionVariables() << '\n';
        std::cout << "Variables\tSize\n";

        std::cout << "Number of Constraints: " << this->NumberOfConstraints() << '\n';
        std::cout << "Constraint\tSize\n";

        std::cout << "Number of Parameters: " << this->NumberOfParameters() << '\n';
        std::cout << "Parameters\tSize\n";

        std::cout << "-----------------------\n";
    }

   private:
    // Program name
    std::string name_;

    // Constrain vector lower bound
    Eigen::VectorXd lbg_;
    // Constrain vector upper bound
    Eigen::VectorXd ubg_;

    // Optimisation vector
    Eigen::VectorXd x_;
};

/**
 * @brief Dense program representation
 *
 */
typedef ProgramBase<Eigen::MatrixXd> Program;

/**
 * @brief Sparse program representation
 *
 */
typedef ProgramBase<Eigen::SparseMatrix<double>> SparseProgram;

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_PROGRAM_H */
