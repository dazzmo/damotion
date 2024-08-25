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
  VLOG(10) << "bounding box constraints";
  for (Binding<BoundingBoxConstraint>& binding :
       getCurrentProgram().g().boundingBox()) {
    // Get bounds
    std::size_t n = binding.x().size();
    Eigen::VectorXd lb(n), ub(n);
    binding.get()->bounds(lb, ub);
    VLOG(10) << "lb: " << lb.transpose();
    VLOG(10) << "ub: " << ub.transpose();
    auto idx = getCurrentProgram().x().getIndices(binding.x());
    context_.lbx(idx) = binding.get()->lb();
    context_.ubx(idx) = binding.get()->ub();
  }
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
  context_.g.setZero();
  context_.H.setZero();

  // Create variable bounds from bounding box constraints
  LOG(INFO) << "bounding box";
  for (Binding<BoundingBoxConstraint>& binding :
       getCurrentProgram().g().boundingBox()) {
    // Get bounds
    std::size_t n = binding.x().size();
    Eigen::VectorXd lb(n), ub(n);
    binding.get()->bounds(lb, ub);
    auto idx = getCurrentProgram().x().getIndices(binding.x());
    context_.lbx(idx) = binding.get()->lb();
    context_.ubx(idx) = binding.get()->ub();
  }

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
    context_.g(indices) = b;
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
    context_.H(indices, indices) = 2 * A;
    context_.g(indices) = b;
  }

  /** Linear constraints **/
  LOG(INFO) << "linear constraints";
  context_.A.setZero();
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
    context_.A.middleRows(cnt, binding.get()->size())(Eigen::all, indices) = A;

    // Adapt bounds for the linear constraints
    context_.ubA.middleRows(cnt, binding.get()->size()) =
        binding.get()->ub() - b;
    context_.lbA.middleRows(cnt, binding.get()->size()) =
        binding.get()->lb() - b;

    // Increase constraint index
    cnt += binding.get()->size();
  }

  // Show the problem coefficients
  VLOG(10) << "H = " << context_.H;
  VLOG(10) << "g = " << context_.g;
  VLOG(10) << "A = " << context_.A;
  VLOG(10) << "lbA = " << context_.lbA;
  VLOG(10) << "ubA = " << context_.ubA;
  VLOG(10) << "lbx = " << context_.lbx;
  VLOG(10) << "ubx = " << context_.ubx;

  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      RowMajorMatrixXd;

  RowMajorMatrixXd H = Eigen::Map<RowMajorMatrixXd>(
      context_.H.data(), context_.H.rows(), context_.H.cols());
  RowMajorMatrixXd A = Eigen::Map<RowMajorMatrixXd>(
      context_.A.data(), context_.A.rows(), context_.A.cols());

  // Solve
  // todo - make this a parameter
  int nWSR = 100;
  if (first_solve_) {
    // Initialise the program and solve it
    qp_->init(H.data(), context_.g.data(), A.data(), context_.lbx.data(),
              context_.ubx.data(), context_.lbA.data(), context_.ubA.data(),
              nWSR);
    first_solve_ = false;
  } else {
    // Use previous solution to hot-start the program
    qp_->hotstart(H.data(), context_.g.data(), A.data(), context_.lbx.data(),
                  context_.ubx.data(), context_.lbA.data(), context_.ubA.data(),
                  nWSR);
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
    qp_->getPrimalSolution(context_.primal.data());
  }
  return context_.primal;
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion