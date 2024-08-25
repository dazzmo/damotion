#ifndef OPTIMISATION_PROGRAM_H
#define OPTIMISATION_PROGRAM_H

#include <casadi/casadi.hpp>

#include "damotion/casadi/codegen.hpp"
#include "damotion/casadi/eigen.hpp"
#include "damotion/casadi/function.hpp"
#include "damotion/core/logging.hpp"
#include "damotion/core/profiler.hpp"
#include "damotion/optimisation/binding.h"
#include "damotion/optimisation/constraints.hpp"
#include "damotion/optimisation/costs.hpp"

namespace damotion {
namespace optimisation {

// Forward declaration of SolverBase
namespace solvers {
class SolverBase;
}

/**
 * @brief Vector of constraint bindings for a system of the form \f$ c(x, p) =
 * [c_0(x, p), c_1(x, p), \hdots , c_n(x, p)]
 *
 */
class ConstraintVector {
 public:
  using Index = std::size_t;

  ConstraintVector() : sz_(0) {}
  ~ConstraintVector() {}

  const Index &size() const { return sz_; }

  /**
   * @brief Add a generic constraint to the program that uses the variables
   * and parameters given by x and p respectively.
   *
   * @param c
   * @param x
   * @param p
   * @return Binding<Constraint>
   */
  Binding<Constraint> add(const Constraint::SharedPtr &con,
                          const symbolic::Vector &x,
                          const symbolic::Vector &p) {
    // Create a binding for the constraint
    constraints_.push_back(Binding<Constraint>(con, x, p));
    sz_ += con->size();
    return constraints_.back();
  }

  /**
   * @brief Add a generic constraint to the program that uses the variables
   * and parameters given by x and p respectively.
   *
   * @param con
   * @param x
   * @param p
   * @return Binding<LinearConstraint>
   */
  Binding<LinearConstraint> add(const LinearConstraint::SharedPtr &con,
                                const symbolic::Vector &x,
                                const symbolic::Vector &p) {
    linear_.push_back(Binding<LinearConstraint>(con, x, p));
    sz_ += con->size();
    return linear_.back();
  }

  Binding<BoundingBoxConstraint> add(
      const BoundingBoxConstraint::SharedPtr &con, const symbolic::Vector &x,
      const symbolic::Vector &p) {
    bounding_box_.push_back(Binding<BoundingBoxConstraint>(con, x, p));
    return bounding_box_.back();
  }

  std::vector<Binding<LinearConstraint>> &linear() { return linear_; }
  std::vector<Binding<BoundingBoxConstraint>> &boundingBox() {
    return bounding_box_;
  }

  /**
   * @brief Provides all constraints in the vector, excluding bounding box
   * constraints.
   *
   * @return std::vector<Binding<Constraint>>
   */
  std::vector<Binding<Constraint>> all() const {
    std::vector<Binding<Constraint>> constraints;
    constraints.insert(constraints.begin(), constraints_.begin(),
                       constraints_.end());
    constraints.insert(constraints.begin(), linear_.begin(), linear_.end());
    // Return vector of all constraints
    return constraints;
  }

  /**
   * @brief Computes the vector bounds from the available bounding box
   * constraints using ordering provided by the variable vector.
   *
   * @param lb
   * @param ub
   */
  void boundingBoxBounds(const Eigen::Ref<Eigen::VectorXd> &lb,
                         const Eigen::Ref<Eigen::VectorXd> &ub,
                         const symbolic::VariableVector &var) {
    VLOG(10) << "ConstraintVector::boundingBoxBounds";
    // For each bounding box constraint
    for (Binding<BoundingBoxConstraint> &binding : boundingBox()) {
      // Get bounds
      std::size_t n = binding.x().size();
      Eigen::VectorXd lb(n), ub(n);
      binding.get()->bounds(lb, ub);

      VLOG(10) << "lb: " << lb.transpose();
      VLOG(10) << "ub: " << ub.transpose();

      auto idx = var.getIndices(binding.x());
      lb(idx) = binding.get()->lb();
      ub(idx) = binding.get()->ub();
    }
  }

  void constraintBounds(Eigen::Ref<Eigen::VectorXd> lb,
                        Eigen::Ref<Eigen::VectorXd> ub) {
    VLOG(10) << "ConstraintVector::constraintBounds";
    // For each bounding box constraint
    std::size_t idx = 0;
    for (Binding<Constraint> &binding : all()) {
      // Get bounds
      std::size_t sz = binding.get()->size();
      VLOG(10) << "lb: " << binding.get()->lb().transpose();
      VLOG(10) << "ub: " << binding.get()->ub().transpose();

      lb.middleRows(idx, sz) = binding.get()->lb();
      ub.middleRows(idx, sz) = binding.get()->ub();
      idx += sz;
    }
  }

  // TODO - Ordering of the constraints

 private:
  Index sz_;

  // Constraint collections
  std::vector<Binding<Constraint>> constraints_;
  std::vector<Binding<LinearConstraint>> linear_;
  std::vector<Binding<BoundingBoxConstraint>> bounding_box_;
};

std::ostream &operator<<(std::ostream &os, const ConstraintVector &cv);

/**
 * @brief Construct the dense constraint Jacobian for the constraint vector g
 * with values x and the ordering of the variable vector v.
 *
 * @param x
 * @param lam
 * @param g
 * @param v
 * @return Eigen::MatrixXd
 */
Eigen::MatrixXd constraintJacobian(const Eigen::VectorXd &x,
                                   const ConstraintVector &g,
                                   const symbolic::VariableVector &v);

Eigen::MatrixXd constraintHessian(const Eigen::VectorXd &x,
                                  const Eigen::VectorXd &lam,
                                  const ConstraintVector &g,
                                  const symbolic::VariableVector &v);

class ObjectiveFunction {
 public:
  Binding<Cost> add(const Cost::SharedPtr &cost, const symbolic::Vector &x,
                    const symbolic::Vector &p) {
    VLOG(10) << "adding generic cost";
    Binding<Cost> binding(cost, x, p);
    costs_.push_back(binding);
    return costs_.back();
  }

  Binding<LinearCost> add(const LinearCost::SharedPtr &cost,
                          const symbolic::Vector &x,
                          const symbolic::Vector &p) {
    VLOG(10) << "adding linear cost";
    linear_costs_.push_back(Binding<LinearCost>(cost, x, p));
    return linear_costs_.back();
  }

  Binding<QuadraticCost> add(const QuadraticCost::SharedPtr &cost,
                             const symbolic::Vector &x,
                             const symbolic::Vector &p) {
    VLOG(10) << "adding quadratic cost";
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
   * @brief Returns a vector of all costs, as bindings to the generic Cost base
   * class.
   *
   * @return std::vector<Binding<Cost>>
   */
  std::vector<Binding<Cost>> all() const {
    std::vector<Binding<Cost>> costs = {};
    costs.insert(costs.begin(), costs_.begin(), costs_.end());
    costs.insert(costs.begin(), linear_costs_.begin(), linear_costs_.end());
    costs.insert(costs.begin(), quadratic_costs_.begin(),
                 quadratic_costs_.end());

    // Return vector of all costs
    return costs;
  }

 private:
  // Cost collections
  std::vector<Binding<Cost>> costs_;
  std::vector<Binding<LinearCost>> linear_costs_;
  std::vector<Binding<QuadraticCost>> quadratic_costs_;
};

/**
 * @brief Construct the dense objective Hessian for the objective function f
 * with values x and the ordering of the variable vector v.
 *
 * @param x
 * @param g
 * @param v
 * @return Eigen::MatrixXd
 */
Eigen::MatrixXd objectiveHessian(const Eigen::VectorXd &x,
                                 const ObjectiveFunction &f,
                                 const symbolic::VariableVector &v);

std::ostream &operator<<(std::ostream &os, const ObjectiveFunction &obj);

/**
 * @brief Represents a generic mathematical program of the form \f$ \min f(x, p)
 * s.t. g_l \le g(x) \le q_u, x_l \le x \le x_u \f$
 *
 */
class MathematicalProgram {
 public:
  friend class solvers::SolverBase;

  using Index = std::size_t;
  using String = std::string;

  MathematicalProgram() = default;

  MathematicalProgram(const String &name) : name_(name) {}

  /**
   * @brief Name of the program
   *
   * @return const String&
   */
  const String &name() const { return name_; }

  /**
   * @brief Objective function for the mathematical program
   *
   * @return ObjectiveFunction&
   */
  ObjectiveFunction &f() { return f_; }

  /**
   * @brief Constraint vector for the program
   *
   * @return ConstraintVector&
   */
  ConstraintVector &g() { return g_; }

  /**
   * @brief Decision variables of the mathematical program
   *
   * @return symbolic::VariableVector&
   */
  symbolic::VariableVector &x() { return x_; }

  /**
   * @brief Parameters for the mathematical program
   *
   * @return symbolic::VariableVector&
   */
  symbolic::VariableVector &p() { return p_; }

 private:
  // todo - Consider programs with different spaces for their state and
  // todo - derivative (e.g. quaternions)

  // Name
  String name_;

  // Optimisation vector
  symbolic::VariableVector x_;
  // Parameter vector
  symbolic::VariableVector p_;

  // Objective function
  ObjectiveFunction f_;

  // Constraint vector
  ConstraintVector g_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* OPTIMISATION_PROGRAM_H */
