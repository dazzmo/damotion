#ifndef OPTIMISATION_PROGRAM_H
#define OPTIMISATION_PROGRAM_H

#include <casadi/casadi.hpp>

#include "damotion/casadi/codegen.h"
#include "damotion/casadi/eigen.h"
#include "damotion/casadi/function.hpp"
#include "damotion/core/logging.hpp"
#include "damotion/core/profiler.hpp"
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
  Binding<Cost> addCost(const std::shared_ptr<Cost> &cost,
                        const sym::VariableRefVector &x,
                        const sym::VariableRefVector &p) {
    Binding<Cost> binding(cost, x, p);
    costs_.push_back(binding);
    return costs_.back();
  }

  Binding<LinearCost> addLinearCost(const std::shared_ptr<LinearCost> &cost,
                                    const sym::VariableRefVector &x,
                                    const sym::VariableRefVector &p) {
    linear_costs_.push_back(Binding<LinearCost>(cost, x, p));
    return linear_costs_.back();
  }

  Binding<QuadraticCost> addQuadraticCost(
      const std::shared_ptr<QuadraticCost> &cost,
      const sym::VariableRefVector &x, const sym::VariableRefVector &p) {
    quadratic_costs_.push_back(Binding<QuadraticCost>(cost, x, p));
    return quadratic_costs_.back();
  }

  std::vector<Binding<Cost>> getAllCostBindings() {
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
  std::vector<Binding<LinearCost>> &getLinearCostBindings() {
    // Create constraints
    return linear_costs_;
  }

  /**
   * @brief Get the vector of current QuadraticCost objects within the
   * program.
   *
   * @return std::vector<Binding<QuadraticCost>>&
   */
  std::vector<Binding<QuadraticCost>> &getQuadraticCostBindings() {
    // Create constraints
    return quadratic_costs_;
  }

  /**
   * @brief Prints the current set of costs for the program to the
   * screen
   *
   */
  void listCosts() {
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
  const int &numberOfConstraints() const { return n_constraints_; }

  /**
   * @brief Add a constraint to the program that uses the variables
   * and parameters given by x and p respectively.
   *
   * @param c
   * @param x
   * @param p
   * @return Binding<Constraint>
   */
  Binding<Constraint> addConstraint(const std::shared_ptr<Constraint> &con,
                                    const sym::VariableRefVector &x,
                                    const sym::VariableRefVector &p) {
    // Create a binding for the constraint
    constraints_.push_back(Binding<Constraint>(con, x, p));
    n_constraints_ += con->dim();
    return constraints_.back();
  }

  Binding<LinearConstraint> addLinearConstraint(
      const std::shared_ptr<LinearConstraint> &con,
      const sym::VariableRefVector &x, const sym::VariableRefVector &p) {
    linear_constraints_.push_back(Binding<LinearConstraint>(con, x, p));
    n_constraints_ += con->dim();
    return linear_constraints_.back();
  }

  Binding<BoundingBoxConstraint> addBoundingBoxConstraint(
      const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
      const sym::VariableVector &x) {
    std::shared_ptr<BoundingBoxConstraint> con =
        std::make_shared<BoundingBoxConstraint>("", lb, ub);
    bounding_box_constraints_.push_back(
        Binding<BoundingBoxConstraint>(con, {x}));
    return bounding_box_constraints_.back();
  }

  Binding<BoundingBoxConstraint> addBoundingBoxConstraint(
      const double &lb, const double &ub, const sym::VariableVector &x) {
    Eigen::VectorXd lbv(x.size()), ubv(x.size());
    lbv.setConstant(lb);
    ubv.setConstant(ub);
    return addBoundingBoxConstraint(lbv, ubv, x);
  }

  std::vector<Binding<Constraint>> getAllConstraintBindings() {
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
  std::vector<Binding<LinearConstraint>> &getLinearConstraintBindings() {
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
  getBoundingBoxConstraintBindings() {
    // Create constraints
    return bounding_box_constraints_;
  }

  void ListConstraints() {
    std::cout << "----------------------\n";
    std::cout << "Constraint\tSize\tLower Bound\tUpper Bound\n";
    std::cout << "----------------------\n";
    // Get all constraints
    std::vector<Binding<Constraint>> constraints = getAllConstraintBindings();
    for (Binding<Constraint> &b : constraints) {
      std::cout << b.get().name() << "\t[" << b.get().dim() << ",1]\n";
      for (size_t i = 0; i < b.get().dim(); ++i) {
        std::cout << b.get().name() << "_" + std::to_string(i) << "\t\t"
                  << b.get().lowerBound()[i] << "\t" << b.get().upperBound()[i]
                  << "\n";
      }
    }
    for (Binding<BoundingBoxConstraint> &b :
         getBoundingBoxConstraintBindings()) {
      std::cout << b.get().name() << "\t[" << b.get().dim() << ",1]\n";
      for (size_t i = 0; i < b.get().dim(); ++i) {
        std::cout << b.get().name() << "_" + std::to_string(i) << "\t\t"
                  << b.get().lowerBound()[i] << "\t" << b.get().upperBound()[i]
                  << "\n";
      }
    }
  }

  void updateConstraintBoundVectors() {
    lbg_.resize(n_constraints_);
    ubg_.resize(n_constraints_);
    // Set first-in-first out order
    int idx = 0;
    for (Binding<Constraint> &b : getAllConstraintBindings()) {
      lbg_.middleRows(idx, b.get().dim()) = b.get().lowerBound();
      ubg_.middleRows(idx, b.get().dim()) = b.get().upperBound();
      idx += b.get().dim();
    }
  }

  const Eigen::VectorXd &constraintLowerBounds() { return lbg_; }
  const Eigen::VectorXd &constraintUpperBounds() { return ubg_; }

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
class Program {
 public:
  Program() : {
    x_manager_ = std::make_shared<sym::VariableManager>();
    p_manager_ = std::make_shared<sym::VariableManager>();
  }

  ~Program() = default;

  friend class solvers::SolverBase;

  Program(const std::string &name)
      : name_(name), variables_(), parameters_(), costs_(), constriants_() {}

  /**
   * @brief Name of the program
   *
   * @return const std::string&
   */
  const std::string &name() const { return name_; }

  /**
   * @brief Provides the manager for the decision variables within the program.
   *
   * @return sym::VariableManager
   */
  sym::VariableManager &decisionVariables() { return variables_; }

  /**
   * @brief Provides the manager for the parameters within the program.
   *
   * @return sym::VariableManager&
   */
  sym::VariableManager &parameters() { return parameters_; }

  /**
   * @brief Provides the manager for the constraints within the program.
   *
   * @return ConstraintManager&
   */
  ConstraintManager &constraints() { return constraints_; }

  /**
   * @brief Provides the manager for the costs within the program.
   *
   * @return CostManager&
   */
  CostManager &costs() { return costs_; }

  /**
   * @brief Prints a summary of the program, listing number of variables,
   * parameters, constraints and costs
   *
   */
  void printProgramSummary() {
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

  // Managers for components of the program
  sym::VariableManager variables_;
  sym::VariableManager parameters_;
  CostManager costs_;
  ConstraintManager constraints_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* OPTIMISATION_PROGRAM_H */
