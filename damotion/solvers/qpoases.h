#ifndef SOLVERS_QPOASES_H
#define SOLVERS_QPOASES_H

// #ifdef WITH_QPOASES

#include <qpOASES.hpp>

#include "damotion/core/logging.hpp"
#include "damotion/core/profiler.hpp"
#include "damotion/optimisation/program.h"
#include "damotion/solvers/base.h"

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

  QPOASESSolverInstance(MathematicalProgram& prog);

  ~QPOASESSolverInstance();

  void reset();

  void solve();

  /**
   * @brief Get the current return status of the program
   *
   * @return const qpOASES::QProblemStatus&
   */
  qpOASES::QProblemStatus GetProblemStatus() const;

  /**
   * @brief Returns the primal solution of the most recent program, if
   * successful, otherwise returns the last successful primal solution.
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd& getPrimalSolution();

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

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

// #endif /* WITH_QPOASES */
#endif /* SOLVERS_QPOASES_H */
