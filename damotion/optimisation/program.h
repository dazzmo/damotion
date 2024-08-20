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

class MathematicalProgram {
 public:
  friend class solvers::SolverBase;

  using Index = std::size_t;
  using String = std::string;

  MathematicalProgram() = default;

  MathematicalProgram(const String &name)
      : name_(name), nx_(0), ng_(0), np_(0) {}

  /**
   * @brief Name of the program
   *
   * @return const String&
   */
  const String &name() const { return name_; }

  const Index &nx() const { return nx_; }
  const Index &ng() const { return ng_; }
  const Index &np() const { return np_; }

  Binding<Cost> addCost(const Cost::SharedPtr &cost, const sym::Vector &x,
                        const sym::Vector &p) {
    Binding<Cost> binding(cost, x, p);
    costs_.push_back(binding);
    return costs_.back();
  }

  Binding<LinearCost> addLinearCost(const LinearCost::SharedPtr &cost,
                                    const sym::Vector &x,
                                    const sym::Vector &p) {
    linear_costs_.push_back(Binding<LinearCost>(cost, x, p));
    return linear_costs_.back();
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

  Binding<QuadraticCost> addQuadraticCost(const QuadraticCost::SharedPtr &cost,
                                          const sym::Vector &x,
                                          const sym::Vector &p) {
    quadratic_costs_.push_back(Binding<QuadraticCost>(cost, x, p));
    return quadratic_costs_.back();
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
   * @brief Returns a vector of all costs, as bindings to the generic Cost base
   * class.
   *
   * @return std::vector<Binding<Cost>>
   */
  std::vector<Binding<Cost>> getAllCostBindings() {
    std::vector<Binding<Cost>> costs = {};
    costs.insert(costs.begin(), costs_.begin(), costs_.end());
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
  void listCosts() {
    std::cout << "----------------------\n";
    std::cout << "Cost\n";
    std::cout << "----------------------\n";
    std::vector<Binding<Cost>> costs = GetAllCostBindings();
    for (Binding<Cost> &b : costs) {
      std::cout << b.Get().name() << '\n';
    }
  }

  /**
   * @brief Add a constraint to the program that uses the variables
   * and parameters given by x and p respectively.
   *
   * @param c
   * @param x
   * @param p
   * @return Binding<Constraint>
   */
  Binding<Constraint> addConstraint(const Constraint::SharedPtr &con,
                                    const sym::Vector &x,
                                    const sym::Vector &p) {
    // Create a binding for the constraint
    constraints_.push_back(Binding<Constraint>(con, x, p));
    return constraints_.back();
  }

  Binding<LinearConstraint> addLinearConstraint(
      const LinearConstraint::SharedPtr &con, const sym::Vector &x,
      const sym::Vector &p) {
    linear_constraints_.push_back(Binding<LinearConstraint>(con, x, p));
    return linear_constraints_.back();
  }

  // Binding<BoundingBoxConstraint> addBoundingBoxConstraint(
  //     const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
  //     const sym::VariableVector &x) {
  //   std::shared_ptr<BoundingBoxConstraint> con =
  //       std::make_shared<BoundingBoxConstraint>("", lb, ub);
  //   bounding_box_constraints_.push_back(
  //       Binding<BoundingBoxConstraint>(con, {x}));
  //   return bounding_box_constraints_.back();
  // }

  std::vector<Binding<Constraint>> getAllConstraintBindings() {
    std::vector<Binding<Constraint>> constraints;
    constraints.insert(constraints.begin(), constraints_.begin(),
                       constraints_.end());
    constraints.insert(constraints.begin(), linear_constraints_.begin(),
                       linear_constraints_.end());
    // Return vector of all constraints
    return constraints;
  }

  void ListConstraints() {
    std::cout << "----------------------\n";
    std::cout << "Constraint\tSize\tLower Bound\tUpper Bound\n";
    std::cout << "----------------------\n";
    // Get all constraints
    std::vector<Binding<Constraint>> constraints = getAllConstraintBindings();
    for (Binding<Constraint> &b : constraints) {
      std::cout << b.get()->name() << "\t[" << b.get()->dim() << ",1]\n";
      for (size_t i = 0; i < b.get()->dim(); ++i) {
        std::cout << b.get()->name() << "_" + std::to_string(i) << "\t\t"
                  << b.get()->lb()[i] << "\t" << b.get()->ub()[i] << "\n";
      }
    }
    for (Binding<BoundingBoxConstraint> &b :
         getBoundingBoxConstraintBindings()) {
      std::cout << b.get()->name() << "\t[" << b.get()->dim() << ",1]\n";
      for (size_t i = 0; i < b.get()->dim(); ++i) {
        std::cout << b.get()->name() << "_" + std::to_string(i) << "\t\t"
                  << b.get()->lb()[i] << "\t" << b.get()->ub()[i] << "\n";
      }
    }
  }

  /**
   * @brief Decision variables of the mathematical program
   * 
   * @return sym::VariableVector& 
  */
  sym::VariableVector &x() { return x_; }

  /**
   * @brief Parameters for the mathematical program 
   * 
   * @return sym::VariableVector& 
  */
  sym::VariableVector &p() { return p_; }

 private:
  // Consider programs with different spaces for their state and derivative
  // (e.g. quaternions)

  Index ng_;

  String name_;

  // Optimisation vector
  sym::VariableVector x_;

  // Parameter vector
  sym::VariableVector p_;

  // Cost collections
  std::vector<Binding<Cost>> costs_;
  std::vector<Binding<LinearCost>> linear_costs_;
  std::vector<Binding<QuadraticCost>> quadratic_costs_;

  // Constraint collections
  std::vector<Binding<Constraint>> constraints_;
  std::vector<Binding<LinearConstraint>> linear_constraints_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* OPTIMISATION_PROGRAM_H */
