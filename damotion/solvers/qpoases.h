#ifndef SOLVERS_QPOASES_H
#define SOLVERS_QPOASES_H

// #ifdef WITH_QPOASES

#include <qpOASES.hpp>

#include "damotion/common/logging.h"
#include "damotion/common/profiler.h"
#include "damotion/optimisation/program.h"
#include "damotion/solvers/solver.h"

namespace damotion {
namespace optimisation {
namespace solvers {

class qpOASESSolverInstanceBase {};

/**
 * @brief Details for the qpOASES solver
 *
 */
struct qpOASESSolverInfo {
  // Return status for the qpOASES solver
  int returnStatus;
  // Error code for the qpOASES solver
  int errorCode;
  // Whether the solve was successful
  bool success;
};

class QPOASESSolverInstance : public SolverBase {
 public:
  QPOASESSolverInstance() = default;

  QPOASESSolverInstance(Program& prog) : Solver(prog) {
    // Create problem
    int nx = GetCurrentProgram().numberOfDecisionVariables();
    int nc = GetCurrentProgram().NumberOfConstraints();
    qp_ = std::make_unique<qpOASES::SQProblem>(nx, nc);
    lbx_ = Eigen::VectorXd::Zero(nx);
    ubx_ = Eigen::VectorXd::Zero(nx);

    // Create variable bounds from bounding box constraints
    for (Binding<BoundingBoxConstraint<Eigen::MatrixXd>>& binding :
         GetCurrentProgram().GetBoundingBoxConstraintBindings()) {
      // For each variable of the constraint
      const sym::VariableVector& v = binding.x(0);
      for (int i = 0; i < v.size(); ++i) {
        lbx_[GetCurrentProgram().getDecisionVariableIndex(v[i])] =
            binding.Get().lowerBound()[i];
        ubx_[GetCurrentProgram().getDecisionVariableIndex(v[i])] =
            binding.Get().upperBound()[i];
      }
      // Set updated constraint to false
      binding.Get().isUpdated() = false;
    }

    // Constraint bounds
    ubA_ = Eigen::VectorXd::Zero(nc);
    lbA_ = Eigen::VectorXd::Zero(nc);

    H_ = Eigen::MatrixXd::Zero(nx, nx);
    g_ = Eigen::VectorXd::Zero(nx);
  }

  ~QPOASESSolverInstance() {}

  void Reset() {
    // Reset the solver flag
    first_solve_ = true;
  }

  void Solve() {
    common::Profiler profiler("QPOASESSolverInstance::Solve");
    // Number of decision variables in the program
    int nx = GetCurrentProgram().numberOfDecisionVariables();
    int nc = GetCurrentProgram().NumberOfConstraints();

    // Reset costs and gradients
    g_.setZero();
    H_.setZero();

    // Update variable bounds, if any have changed
    for (Binding<BoundingBoxConstraint<Eigen::MatrixXd>>& binding :
         GetCurrentProgram().GetBoundingBoxConstraintBindings()) {
      if (binding.Get().isUpdated()) {
        // For each variable of the constraint
        const sym::VariableVector& v = binding.x(0);
        for (int i = 0; i < v.size(); ++i) {
          lbx_[GetCurrentProgram().getDecisionVariableIndex(v[i])] =
              binding.Get().lowerBound()[i];
          ubx_[GetCurrentProgram().getDecisionVariableIndex(v[i])] =
              binding.Get().upperBound()[i];
        }

        binding.Get().isUpdated() = false;
      }
    }
    // Linear costs
    for (Binding<LinearCost<Eigen::MatrixXd>>& binding :
         GetCurrentProgram().GetLinearCostBindings()) {
      BindingInputData& data = GetBindingInputData(binding);

      // Evaluate the cost
      EvaluateCost(binding, primal_solution_x_, true, true, false);

      // Update the gradient
      InsertVectorAtVariableLocations(g_, binding.Get().c(), binding.x(0),
                                      data.x_continuous[0]);
    }
    // Quadratic costs
    for (Binding<QuadraticCost<Eigen::MatrixXd>>& binding :
         GetCurrentProgram().GetQuadraticCostBindings()) {
      BindingInputData& data = GetBindingInputData(binding);

      // Evaluate the cost
      EvaluateCost(binding, primal_solution_x_, true, true, false);
      // Update the gradient
      InsertVectorAtVariableLocations(g_, binding.Get().b(), binding.x(0),
                                      data.x_continuous[0]);
      // Update the hessian
      InsertHessianAtVariableLocations(
          H_, 2.0 * binding.Get().A(), binding.x(0), binding.x(0),
          data.x_continuous[0], data.x_continuous[0]);
    }
    // Evaluate only the linear constraints of the program
    // Reset constraint jacobian
    constraint_jacobian_cache_.setZero();
    int idx = 0;
    for (const Binding<LinearConstraint<Eigen::MatrixXd>>& binding :
         GetCurrentProgram().GetLinearConstraintBindings()) {
      // Compute the constraints
      EvaluateConstraint(binding, idx, primal_solution_x_, true, true);

      // Adapt bounds for the linear constraints
      ubA_.middleRows(idx, binding.Get().dim()) =
          binding.Get().upperBound() - binding.Get().b();
      lbA_.middleRows(idx, binding.Get().dim()) =
          binding.Get().lowerBound() - binding.Get().b();

      // Increase constraint index
      idx += binding.Get().dim();
    }

    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>
        RowMajorMatrixXd;

    RowMajorMatrixXd H =
        Eigen::Map<RowMajorMatrixXd>(H_.data(), H_.rows(), H_.cols());
    RowMajorMatrixXd A = Eigen::Map<RowMajorMatrixXd>(
        constraint_jacobian_cache_.data(), constraint_jacobian_cache_.rows(),
        constraint_jacobian_cache_.cols());

    // Show the problem coefficients
    VLOG(10) << "H = " << H;
    VLOG(10) << "g = " << g_;
    VLOG(10) << "A = " << A;

    // Solve
    int nWSR = 100;
    if (first_solve_) {
      // Initialise the program and solve it
      qp_->init(H.data(), g_.data(), A.data(), lbx_.data(), ubx_.data(),
                lbA_.data(), ubA_.data(), nWSR);
      first_solve_ = false;
    } else {
      // Use previous solution to hot-start the program
      qp_->hotstart(H.data(), g_.data(), A.data(), lbx_.data(), ubx_.data(),
                    lbA_.data(), ubA_.data(), nWSR);
    }

    // Get primal solution
    n_solves_++;
  }

  /**
   * @brief Get the current return status of the program
   *
   * @return const qpOASES::QProblemStatus&
   */
  qpOASES::QProblemStatus GetProblemStatus() const { return qp_->getStatus(); }

  /**
   * @brief Returns the primal solution of the most recent program, if
   * successful, otherwise returns the last successful primal solution.
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd& GetPrimalSolution() {
    if (GetProblemStatus() == qpOASES::QProblemStatus::QPS_SOLVED) {
      qp_->getPrimalSolution(primal_solution_x_.data());
    }
    return primal_solution_x_;
  }

 private:
  bool first_solve_ = true;
  int n_solves_ = 0;

  // Sparse method
  std::unique_ptr<qpOASES::SQProblem> qp_;

  Eigen::MatrixXd H_;
  Eigen::VectorXd g_;

  Eigen::VectorXd lbx_;
  Eigen::VectorXd ubx_;

  Eigen::VectorXd lbA_;
  Eigen::VectorXd ubA_;
};

class QPOASESSolver {
 public:
  QPOASESSolver() = default;
  ~QPOASESSolver() = default;

  QPOASESSolver(Program& program) {
    // Create new instance
    qp_ = std::make_unique<QPOASESSolverInstance>(program);
  }

  // Update program // TODO - Throw warning if program sizes don't match?
  void UpdateProgram(Program& program) { qp_->UpdateProgram(program); }

  /**
   * @brief Resets the solver
   *
   */
  void Reset() { qp_->Reset(); }

  /**
   * @brief Solves the current program
   *
   */
  void Solve() { qp_->Solve(); }

  const Eigen::VectorXd& GetPrimalSolution() const {
    return qp_->GetPrimalSolution();
  }

 private:
  // Constraint bindings
  std::unique_ptr<QPOASESSolverInstance> qp_;
};

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

// #endif /* WITH_QPOASES */
#endif /* SOLVERS_QPOASES_H */
