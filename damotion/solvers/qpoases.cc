#include "damotion/solvers/qpoases.h"

namespace damotion {
namespace optimisation {
namespace solvers {

QPOASESSolverInstance::QPOASESSolverInstance(MathematicalProgram& prog)
    : SolverBase(prog) {
  // Create problem
  int nx = getCurrentProgram().x().size();
  int ng = getCurrentProgram().g().size();

  qp_ = std::make_unique<qpOASES::SQProblem>(nx, ng);
  lbx_ = Vector::Zero(nx);
  ubx_ = Vector::Zero(nx);

  constexpr double inf = std::numeric_limits<double>::infinity();
  lbx_.setConstant(-inf);
  ubx_.setConstant(inf);

  // // Create variable bounds from bounding box constraints
  // for (Binding<BoundingBoxConstraint<Eigen::MatrixXd>>& binding :
  //      getCurrentProgram().GetBoundingBoxConstraintBindings()) {
  //   // For each variable of the constraint
  //   const sym::VariableVector& v = binding.x(0);
  //   for (int i = 0; i < v.size(); ++i) {
  //     std::size_t idx = getCurrentProgram().x().getIndex(v[i]);
  //     lbx_[idx] = binding.get()->lb()[i];
  //     ubx_[idx] = binding.get()->ub()[i];
  //   }
  // }

  // Constraint bounds
  ubA_ = Eigen::VectorXd::Zero(ng);
  lbA_ = Eigen::VectorXd::Zero(ng);

  H_ = Eigen::MatrixXd::Zero(nx, nx);
  g_ = Eigen::VectorXd::Zero(nx);
  A_ = Eigen::MatrixXd::Zero(ng, nx);
}

QPOASESSolverInstance::~QPOASESSolverInstance() = default;

void QPOASESSolverInstance::reset() {
  // Reset the solver flag
  first_solve_ = true;
}

void QPOASESSolverInstance::solve() {
  Profiler profiler("QPOASESSolverInstance::Solve");
  // Number of decision variables in the program
  int nx = getCurrentProgram().x().size();
  int nc = getCurrentProgram().g().size();

  // Reset costs and gradients
  g_.setZero();
  H_.setZero();

  // // Create variable bounds from bounding box constraints
  // for (Binding<BoundingBoxConstraint<Eigen::MatrixXd>>& binding :
  //      getCurrentProgram().GetBoundingBoxConstraintBindings()) {
  //   // For each variable of the constraint
  //   const sym::VariableVector& v = binding.x(0);
  //   for (int i = 0; i < v.size(); ++i) {
  //     std::size_t idx = getCurrentProgram().x().getIndex(v[i]);
  //     lbx_[idx] = 0.0;  // binding.get()->lowerBound()[i];
  //     ubx_[idx] = 0.0;  // binding.get()->upperBound()[i];
  //   }
  // }

  /** Linear costs **/
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
    g_(indices) = b;
  }

  /** Quadratic costs **/
  for (Binding<QuadraticCost>& binding :
       getCurrentProgram().f().getQuadraticCostBindings()) {
    auto indices = getCurrentProgram().x().getIndices(binding.x());
    // Create coefficients
    Matrix A(binding.x().size(), binding.x().size());
    Vector b(binding.x().size());
    double c = 0.0;
    // Evaluate the cost
    binding.get()->coeffs(A, b, c);
    // Update the hessian
    H_(indices, indices) = 2 * A;
    g_(indices) = b;
  }

  /** Linear constraints **/
  A_.setZero();
  std::size_t cnt = 0;
  for (const Binding<LinearConstraint>& binding :
       getCurrentProgram().g().getLinearConstraintBindings()) {
    // Get coefficients
    Matrix A(binding.get()->size(), binding.x().size());
    Vector b(binding.get()->size());
    // Evaluate the system
    // todo - update parameters
    binding.get()->coeffs(A, b);

    std::cout << A << std::endl;
    std::cout << b << std::endl;

    // Get indices
    auto indices = getCurrentProgram().x().getIndices(binding.x());
    A_.middleRows(cnt, binding.get()->size())(Eigen::all, indices) = A;

    std::cout << A_ << std::endl;
    std::cout << binding.get()->ub() << std::endl;
    std::cout << binding.get()->lb() << std::endl;

    // Adapt bounds for the linear constraints
    ubA_.middleRows(cnt, binding.get()->size()) = binding.get()->ub() - b;
    lbA_.middleRows(cnt, binding.get()->size()) = binding.get()->lb() - b;

    std::cout << ubA_ << std::endl;
    std::cout << lbA_ << std::endl;

    // Increase constraint index
    cnt += binding.get()->size();
  }

  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
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