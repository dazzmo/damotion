#include "damotion/solvers/qpoases.h"

namespace damotion {
namespace optimisation {
namespace solvers {

QPOASESSolverInstance::QPOASESSolverInstance(MathematicalProgram& prog)
    : SolverBase(prog) {
  LOG(INFO) << "QPOASESSolverInstance::QPOASESSolverInstance";
  // Create problem
  int nx = getCurrentProgram().x().size();
  int ng = getCurrentProgram().g().size();

  qp_ = std::make_unique<qpOASES::SQProblem>(nx, ng);

  context_ = QPOASESSolverContext(nx, ng);

  // Create variable bounds from bounding box constraints
  getCurrentProgram().g().boundingBoxBounds(getContext().lbx, getContext().ubx,
                                            getCurrentProgram().x());
}

QPOASESSolverInstance::~QPOASESSolverInstance() = default;

void QPOASESSolverInstance::reset() {
  // Reset the solver flag
  first_solve_ = true;
}

void QPOASESSolverInstance::solve() {
  LOG(INFO) << "QPOASESSolverInstance::solve";
  Profiler profiler("QPOASESSolverInstance::Solve");
  // Number of decision variables in the program
  int nx = getCurrentProgram().x().size();
  int nc = getCurrentProgram().g().size();

  // Reset costs and gradients
  getContext().g.setZero();
  getContext().H.setZero();

  // Create variable bounds from bounding box constraints
  LOG(INFO) << "bounding box";
  getCurrentProgram().g().boundingBoxBounds(getContext().lbx, getContext().ubx,
                                            getCurrentProgram().x());

  /** Linear costs **/
  LOG(INFO) << "linear costs";
  for (const Binding<LinearCost>& binding :
       getCurrentProgram().f().getLinearCostBindings()) {
    // Get coefficients
    Vector b(binding.x().size());
    double c = 0.0;
    // Evaluate the system
    binding.get()->coeffs(b, c);
    VLOG(10) << "b: " << b.transpose();
    VLOG(10) << "c: " << c;
    // Get index
    auto indices = getCurrentProgram().x().getIndices(binding.x());
    // Insert at locations
    getContext().g(indices) = b;
  }

  /** Quadratic costs **/
  LOG(INFO) << "quadratic costs";
  for (Binding<QuadraticCost>& binding :
       getCurrentProgram().f().getQuadraticCostBindings()) {
    auto indices = getCurrentProgram().x().getIndices(binding.x());
    // Create coefficients
    Matrix A(binding.x().size(), binding.x().size());
    Vector b(binding.x().size());
    double c = 0.0;
    // Evaluate the cost
    binding.get()->coeffs(A, b, c);
    VLOG(10) << "A: " << A;
    VLOG(10) << "b: " << b.transpose();
    VLOG(10) << "c: " << c;
    // Update the hessian
    getContext().H(indices, indices) = 2 * A;
    getContext().g(indices) = b;
  }

  /** Linear constraints **/
  LOG(INFO) << "linear constraints";
  getContext().A.setZero();
  std::size_t cnt = 0;
  for (const Binding<LinearConstraint>& binding :
       getCurrentProgram().g().linear()) {
    // Get coefficients
    Matrix A(binding.get()->size(), binding.x().size());
    Vector b(binding.get()->size());
    // Evaluate the system
    // todo - update parameters
    binding.get()->coeffs(A, b);
    VLOG(10) << "A: " << A;
    VLOG(10) << "b: " << b.transpose();
    // Get indices
    auto indices = getCurrentProgram().x().getIndices(binding.x());
    getContext().A.middleRows(cnt, binding.get()->size())(Eigen::all, indices) =
        A;

    // Adapt bounds for the linear constraints
    getContext().ubA.middleRows(cnt, binding.get()->size()) =
        binding.get()->ub() - b;
    getContext().lbA.middleRows(cnt, binding.get()->size()) =
        binding.get()->lb() - b;

    // Increase constraint index
    cnt += binding.get()->size();
  }

  // Show the problem coefficients
  VLOG(10) << "H = " << getContext().H;
  VLOG(10) << "g = " << getContext().g;
  VLOG(10) << "A = " << getContext().A;
  VLOG(10) << "lbA = " << getContext().lbA;
  VLOG(10) << "ubA = " << getContext().ubA;
  VLOG(10) << "lbx = " << getContext().lbx;
  VLOG(10) << "ubx = " << getContext().ubx;

  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      RowMajorMatrixXd;

  RowMajorMatrixXd H = Eigen::Map<RowMajorMatrixXd>(
      getContext().H.data(), getContext().H.rows(), getContext().H.cols());
  RowMajorMatrixXd A = Eigen::Map<RowMajorMatrixXd>(
      getContext().A.data(), getContext().A.rows(), getContext().A.cols());

  // Solve
  // todo - make this a parameter
  int nWSR = 100;
  if (first_solve_) {
    // Initialise the program and solve it
    qp_->init(H.data(), getContext().g.data(), A.data(),
              getContext().lbx.data(), getContext().ubx.data(),
              getContext().lbA.data(), getContext().ubA.data(), nWSR);
    first_solve_ = false;
  } else {
    // Use previous solution to hot-start the program
    qp_->hotstart(H.data(), getContext().g.data(), A.data(),
                  getContext().lbx.data(), getContext().ubx.data(),
                  getContext().lbA.data(), getContext().ubA.data(), nWSR);
  }

  // Get primal solution
  n_solves_++;
}

/**
 * @brief Get the current return status of the program
 *
 * @return const qpOASES::QProblemStatus&
 */
qpOASES::QProblemStatus QPOASESSolverInstance::GetProblemStatus() const {
  return qp_->getStatus();
}

/**
 * @brief Returns the primal solution of the most recent program, if
 * successful, otherwise returns the last successful primal solution.
 *
 * @return const Eigen::VectorXd&
 */
const Eigen::VectorXd& QPOASESSolverInstance::getPrimalSolution() {
  if (GetProblemStatus() == qpOASES::QProblemStatus::QPS_SOLVED) {
    qp_->getPrimalSolution(getContext().primal.data());
  }
  return getContext().primal;
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion