#ifndef OPTIMISATION_PROGRAM_H
#define OPTIMISATION_PROGRAM_H

#include <casadi/casadi.hpp>

#include "damotion/casadi/codegen.h"
#include "damotion/casadi/eigen.h"
#include "damotion/casadi/function.h"
#include "damotion/common/logging.h"
#include "damotion/common/profiler.h"
#include "damotion/optimisation/binding.h"
#include "damotion/optimisation/constraints/constraints.h"
#include "damotion/optimisation/costs/costs.h"
#include "damotion/optimisation/fwd.h"

namespace damotion {
namespace optimisation {

// Forward declaration of SolverBase
namespace solvers {
class SolverBase;
}

class CostManager {
 public:
  CostManager() = default;
  ~CostManager() = default;

  /**
   * @brief Adds a generic cost
   *
   * @param cost
   * @param x
   * @param p
   * @return Binding<Cost>
   */
  Binding<Cost> AddCost(const std::shared_ptr<Cost> &cost,
                        const sym::VariableRefVector &x,
                        const sym::VariableRefVector &p) {
    Binding<Cost> binding(cost, x, p);
    costs_.push_back(binding);
    return costs_.back();
  }

  Binding<LinearCost> AddLinearCost(const std::shared_ptr<LinearCost> &cost,
                                    const sym::VariableRefVector &x,
                                    const sym::VariableRefVector &p) {
    linear_costs_.push_back(Binding<LinearCost>(cost, x, p));
    return linear_costs_.back();
  }

  Binding<QuadraticCost> AddQuadraticCost(
      const std::shared_ptr<QuadraticCost> &cost,
      const sym::VariableRefVector &x, const sym::VariableRefVector &p) {
    quadratic_costs_.push_back(Binding<QuadraticCost>(cost, x, p));
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
   * @brief Get the vector of current LinearCost objects within the
   * program.
   *
   * @return std::vector<Binding<LinearCost>>&
   */
  std::vector<Binding<LinearCost>> &GetLinearCostBindings() {
    // Create constraints
    return linear_costs_;
  }

  /**
   * @brief Get the vector of current QuadraticCost objects within the
   * program.
   *
   * @return std::vector<Binding<QuadraticCost>>&
   */
  std::vector<Binding<QuadraticCost>> &GetQuadraticCostBindings() {
    // Create constraints
    return quadratic_costs_;
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
  std::vector<Binding<LinearCost>> linear_costs_;
  std::vector<Binding<QuadraticCost>> quadratic_costs_;
};

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
   * @return Binding<Constraint>
   */
  Binding<Constraint> AddConstraint(const std::shared_ptr<Constraint> &con,
                                    const sym::VariableRefVector &x,
                                    const sym::VariableRefVector &p) {
    // Create a binding for the constraint
    constraints_.push_back(Binding<Constraint>(con, x, p));
    n_constraints_ += con->Dimension();
    return constraints_.back();
  }

  Binding<LinearConstraint> AddLinearConstraint(
      const std::shared_ptr<LinearConstraint> &con,
      const sym::VariableRefVector &x, const sym::VariableRefVector &p) {
    linear_constraints_.push_back(Binding<LinearConstraint>(con, x, p));
    n_constraints_ += con->Dimension();
    return linear_constraints_.back();
  }

  Binding<BoundingBoxConstraint> AddBoundingBoxConstraint(
      const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
      const sym::VariableVector &x) {
    std::shared_ptr<BoundingBoxConstraint> con =
        std::make_shared<BoundingBoxConstraint>("", lb, ub);
    bounding_box_constraints_.push_back(
        Binding<BoundingBoxConstraint>(con, {x}));
    return bounding_box_constraints_.back();
  }

  Binding<BoundingBoxConstraint> AddBoundingBoxConstraint(
      const double &lb, const double &ub, const sym::VariableVector &x) {
    Eigen::VectorXd lbv(x.size()), ubv(x.size());
    lbv.setConstant(lb);
    ubv.setConstant(ub);
    return AddBoundingBoxConstraint(lbv, ubv, x);
  }

  std::vector<Binding<Constraint>> GetAllConstraintBindings() {
    std::vector<Binding<Constraint>> constraints;
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

  void ListConstraints() {
    std::cout << "----------------------\n";
    std::cout << "Constraint\tSize\tLower Bound\tUpper Bound\n";
    std::cout << "----------------------\n";
    // Get all constraints
    std::vector<Binding<Constraint>> constraints = GetAllConstraintBindings();
    for (Binding<Constraint> &b : constraints) {
      std::cout << b.Get().name() << "\t[" << b.Get().Dimension() << ",1]\n";
      for (int i = 0; i < b.Get().Dimension(); ++i) {
        std::cout << b.Get().name() << "_" + std::to_string(i) << "\t\t"
                  << b.Get().lowerBound()[i] << "\t" << b.Get().upperBound()[i]
                  << "\n";
      }
    }
    for (Binding<BoundingBoxConstraint> &b :
         GetBoundingBoxConstraintBindings()) {
      std::cout << b.Get().name() << "\t[" << b.Get().Dimension() << ",1]\n";
      for (int i = 0; i < b.Get().Dimension(); ++i) {
        std::cout << b.Get().name() << "_" + std::to_string(i) << "\t\t"
                  << b.Get().lowerBound()[i] << "\t" << b.Get().upperBound()[i]
                  << "\n";
      }
    }
  }

  void UpdateConstraintBoundVectors() {
    lbg_.resize(n_constraints_);
    ubg_.resize(n_constraints_);
    // Set first-in-first out order
    int idx = 0;
    for (Binding<Constraint> &b : GetAllConstraintBindings()) {
      lbg_.middleRows(idx, b.Get().Dimension()) = b.Get().lowerBound();
      ubg_.middleRows(idx, b.Get().Dimension()) = b.Get().upperBound();
      idx += b.Get().Dimension();
    }
  }

  const Eigen::VectorXd &ConstraintlowerBounds() { return lbg_; }
  const Eigen::VectorXd &ConstraintupperBounds() { return ubg_; }

 private:
  // Number of constraints
  int n_constraints_;

  std::vector<Binding<Constraint>> constraints_;
  std::vector<Binding<LinearConstraint>> linear_constraints_;
  std::vector<Binding<BoundingBoxConstraint>> bounding_box_constraints_;

  // Constraint vector lower bound
  Eigen::VectorXd lbg_;
  // Constraint vector upper bound
  Eigen::VectorXd ubg_;
};

/**
 * @brief The program class creates mathematical programs.
 * It uses a block structure to allow greater speed in accessing data, thus
 * optimisation variables are grouped in sets, which can be as small as a single
 * unit.
 *
 */
class Program : public CostManager, public ConstraintManager {
 public:
  Program() {
    x_manager_ = std::make_shared<sym::VariableManager>();
    p_manager_ = std::make_shared<sym::VariableManager>();
  }

  ~Program() = default;

  friend class solvers::SolverBase;

  Program(const std::string &name) : name_(name) {}

  /**
   *
   * Decision Variable Manager
   *
   */

  /**
   * @brief Number of decision variables currently in the program
   *
   * @return const int&
   */
  const int &NumberOfDecisionVariables() const;

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
  void AddDecisionVariables(const Eigen::Ref<const sym::VariableMatrix> &var);

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
   * @brief Returns a vector of indices for the position of each entry in v in
   * the current decision variable vector.
   *
   * @param v
   * @return std::vector<int>
   */
  std::vector<int> GetDecisionVariableIndices(const sym::VariableVector &v);

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
   * @brief Determines whether a vector of variables var is continuous within
   * the optimisation vector of the program.
   *
   * @param var
   * @return true
   * @return false
   */
  bool IsContinuousInDecisionVariableVector(const sym::VariableVector &var);
  /**
   * @brief Updates the decision variable bound vectors with all the current
   * values set for the decision variables.
   *
   */
  void UpdateDecisionVariableBoundVectors();

  /**
   * @brief Updates the initial value vector for the decision variables with all
   * the current values set for the decision variables.
   *
   */
  void UpdateDecisionVariableInitialValueVector();

  /**
   * @brief Vector of initial values for the decision variables within the
   * program.
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &DecisionVariableInitialValues() const;

  /**
   * @brief Upper bound for decision variables within the current program.
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &DecisionVariableupperBounds() const;

  /**
   * @brief Upper bound for decision variables within the current program.
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &DecisionVariablelowerBounds() const;

  void SetDecisionVariableBounds(const sym::Variable &v, const double &lb,
                                 const double &ub);
  void SetDecisionVariableBounds(const sym::VariableVector &v,
                                 const Eigen::VectorXd &lb,
                                 const Eigen::VectorXd &ub);

  void SetDecisionVariableInitialValue(const sym::Variable &v,
                                       const double &x0);
  void SetDecisionVariableInitialValue(const sym::VariableVector &v,
                                       const Eigen::VectorXd &x0);

  /**
   * @brief Prints the current set of parameters for the program to the
   * screen
   *
   */
  void ListDecisionVariables();

  /**
   *
   * Parameters
   *
   */

  /**
   * @brief Number of parameters currently in the program
   *
   * @return const int&
   */
  const int &NumberOfParameters() const;

  /**
   * @brief Adds a parameter
   *
   * @param par
   */
  void AddParameter(const sym::Variable &par);
  /**
   * @brief Add parameters
   *
   * @param par
   */
  void AddParameters(const Eigen::Ref<const sym::VariableMatrix> &par);

  /**
   * @brief Removes variables currently considered by the program.
   *
   * @param par
   */
  void RemoveParameters(const Eigen::Ref<sym::VariableMatrix> &par);

  /**
   * @brief Whether a variable var is a parameter within the program
   *
   * @param par
   * @return true
   * @return false
   */
  bool IsParameter(const sym::Variable &par);

  /**
   * @brief Returns the index of the given variable within the created
   * optimisation vector
   *
   * @param p
   * @return int
   */
  int GetParameterIndex(const sym::Variable &p);

  /**
   * @brief Returns a vector of indices for the position of each entry in v in
   * the current parameter vector.
   *
   * @param p
   * @return std::vector<int>
   */
  std::vector<int> GetParameterIndices(const sym::VariableVector &v);

  /**
   * @brief Set the vector of parameters to the default ordering of
   * variables (ordered by when they were added)
   *
   */
  void SetParameterVector();

  /**
   * @brief Sets the optimisation vector with the given ordering of variables
   *
   * @param par
   */
  bool SetParameterVector(const Eigen::Ref<sym::VariableVector> &par);

  /**
   * @brief Determines whether a vector of variables var is continuous within
   * the optimisation vector of the program.
   *
   * @param par
   * @return true
   * @return false
   */
  bool IsContinuousInParameterVector(const sym::VariableVector &par);

  /**
   * @brief Updates the initial value vector for the parameters with all
   * the current values set for the parameters.
   *
   */
  void UpdateParameterValueVector();

  /**
   * @brief Vector of initial values for the parameters within the
   * program.
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &ParameterValues() const;

  void SetParameterValue(const sym::Variable &p, const double &p0);
  void SetParameterValue(const sym::VariableVector &p,
                         const Eigen::VectorXd &p0);

  /**
   * @brief Prints the current set of parameters for the program to the
   * screen
   *
   */
  void ListParameters();

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

  /**
   * @brief Prints a summary of the program, listing number of variables,
   * parameters, constraints and costs
   *
   */
  void PrintProgramSummary() {
    std::cout << "-----------------------\n";
    std::cout << "Program Name: " << name() << '\n';
    std::cout << "Number of Decision Variables: "
              << this->x_manager_->NumberOfVariables() << '\n';
    std::cout << "Variables\tSize\n";

    std::cout << "Number of Constraints: " << this->NumberOfConstraints()
              << '\n';
    std::cout << "Constraint\tSize\n";

    std::cout << "Number of Parameters: "
              << this->p_manager_->NumberOfVariables() << '\n';
    std::cout << "Parameters\tSize\n";

    std::cout << "-----------------------\n";
  }

 private:
  // Program name
  std::string name_;

  // Optimisation vector
  Eigen::VectorXd x_;

  sym::VariableManager::SharedPtr x_manager_;
  sym::VariableManager::SharedPtr p_manager_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* OPTIMISATION_PROGRAM_H */
