#ifndef SOLVERS_QPOASES_H
#define SOLVERS_QPOASES_H

// #ifdef WITH_QPOASES

#include <qpOASES.hpp>

#include "damotion/core/logging.h"
#include "damotion/core/profiler.h"
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
  // Number of working sets performed
  int nWSR;
  // Whether the solve was successful
  bool success;
};

class QPOASESSolverInstance : public SolverBase {
 public:
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;

  QPOASESSolverInstance() = default;

  QPOASESSolverInstance(Program& prog) : Solver(prog) {
    // Create problem
    int nx = getCurrentProgram().x().size();
    int ng = getCurrentProgram().g().size();

    qp_ = std::make_unique<qpOASES::SQProblem>(nx, nc);
    lbx_ = Vector::Zero(nx);
    ubx_ = Vector::Zero(nx);

    // Create variable bounds from bounding box constraints
    for (Binding<BoundingBoxConstraint<Eigen::MatrixXd>>& binding :
         getCurrentProgram().GetBoundingBoxConstraintBindings()) {
      // For each variable of the constraint
      const sym::VariableVector& v = binding.x(0);
      for (int i = 0; i < v.size(); ++i) {
        std::size_t idx = getCurrentProgram().x().getIndex(v[i]);
        lbx_[idx] = 0.0;  // binding.get()->lowerBound()[i];
        ubx_[idx] = 0.0;  // binding.get()->upperBound()[i];
      }
    }

    // Constraint bounds
    ubA_ = Eigen::VectorXd::Zero(nc);
    lbA_ = Eigen::VectorXd::Zero(nc);

    H_ = Eigen::MatrixXd::Zero(nx, nx);
    g_ = Eigen::VectorXd::Zero(nx);
  }

  ~QPOASESSolverInstance() {}

  void reset() {
    // Reset the solver flag
    first_solve_ = true;
  }

  void solve() {
    core::Profiler profiler("QPOASESSolverInstance::Solve");
    // Number of decision variables in the program
    int nx = getCurrentProgram().x().size();
    int nc = getCurrentProgram().g().size();

    // Reset costs and gradients
    g_.setZero();
    H_.setZero();

    // Create variable bounds from bounding box constraints
    for (Binding<BoundingBoxConstraint<Eigen::MatrixXd>>& binding :
         getCurrentProgram().GetBoundingBoxConstraintBindings()) {
      // For each variable of the constraint
      const sym::VariableVector& v = binding.x(0);
      for (int i = 0; i < v.size(); ++i) {
        std::size_t idx = getCurrentProgram().x().getIndex(v[i]);
        lbx_[idx] = 0.0;  // binding.get()->lowerBound()[i];
        ubx_[idx] = 0.0;  // binding.get()->upperBound()[i];
      }
    }

    // Linear costs
    for (const Binding<LinearCost>& binding :
         getCurrentProgram().f().getLinearCostBindings()) {
      // Get coefficients
      Vector b(binding.x().size());
      double c = 0.0;
      // Evaluate the system
      binding.get()->coeffs(b, c);

      // Get index
      auto indices = getCurrentProgram().x().getIndices(binding.x());
      // Insert at locations
      // todo - gradient_(indices) << b;
    }
    // Quadratic costs
    for (Binding<QuadraticCost>& binding :
         getCurrentProgram().f().getQuadraticCostBindings()) {
      BindingInputData& data = GetBindingInputData(binding);

      auto indices = getCurrentProgram().x().getIndices(binding.x());
      Matrix A(binding.x().size(), binding.x().size());
      Vector b(binding.x().size());
      double c = 0.0;
      // Evaluate the cost
      binding.get()->coeffs(A, b, c);
      // Update the hessian
      H_(indices, indices) = 2 * A;
      g_(indices) = b;
    }
    // Evaluate only the linear constraints of the program
    // Reset constraint jacobian
    A_.setZero();
    int cnt = 0;
    for (const Binding<LinearConstraint>& binding :
         getCurrentProgram().g().getLinearConstraintBindings()) {
      // Get coefficients
      Matrix A(binding.get()->size(), binding.x().size());
      Vector b(binding.get()->size());
      // Evaluate the system
      // todo - update parameters
      binding.get()->coeffs(A, b);
      // Get indices
      auto indices = getCurrentProgram().x().getIndices(binding.x());

      A_.middleRows(cnt, binding.get()->size())(indices) = A;
      A_.middleRows(cnt, binding.get()->size()) = b;

      // Adapt bounds for the linear constraints
      ubA_.middleRows(cnt, binding.get()->size()) = binding.get()->ub() - b;
      lbA_.middleRows(cnt, binding.get()->size()) = binding.get()->lb() - b;

      // Increase constraint index
      cnt += binding.get()->size();
    }

    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>
        RowMajorMatrixXd;

    RowMajorMatrixXd H =
        Eigen::Map<RowMajorMatrixXd>(H_.data(), H_.rows(), H_.cols());
    RowMajorMatrixXd A =
        Eigen::Map<RowMajorMatrixXd>(A_.data(), A_.rows(), A_.cols());

    // Show the problem coefficients
    VLOG(10) << "H = " << H;
    VLOG(10) << "g = " << g_;
    VLOG(10) << "A = " << A;

    // Solve
    // todo - make this a parameter
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

    // Get solver information

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

  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;

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
