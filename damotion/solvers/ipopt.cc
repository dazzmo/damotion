#include "damotion/solvers/ipopt.h"

namespace damotion {
namespace optimisation {
namespace solvers {

IpoptSolverInstance::IpoptSolverInstance(SparseProgram& prog)
    : Ipopt::TNLP(), SparseSolver(prog) {}

bool IpoptSolverInstance::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                                       Index& nnz_h_lag,
                                       IndexStyleEnum& index_style) {
  VLOG(10) << "get_nlp_info()";
  n = getCurrentProgram().x().size();
  m = getCurrentProgram().g().size();
  // TODO - Determine the sparsity of the hessians and jacobians

  // Estimate sparsity pattern of the systems
  // Use a series of random vectors to estimate
  // todo - if too large, split it up and estimate in smaller bursts

  Eigen::MatrixXd J;
  J = Eigen::MatrixXd::Zero(m, n);
  for (Index i = 0; i < 5; ++i) {
    Eigen::VectorXd in = Eigen::VectorXd::Random(n);
    J +=
        constraintJacobian(x, getCurrentProgram().g(), getCurrentProgram().x());
  }

  // Estimate sparse jacobian
  Eigen::SparseMatrix<double> jac_sparse = J.sparseView();

  nnz_jac_g = jac_sparse().nonZeros();
  nnz_h_lag = GetSparseLagrangianHessian().nonZeros();

  index_style = TNLP::C_STYLE;

  return true;
}

bool IpoptSolverInstance::eval_f(Index n, const Number* x, bool new_x,
                                 Number& obj_value) {
  core::Profiler profiler("IpoptSolverInstance::eval_f");
  VLOG(10) << "eval_f()";

  if (new_x) {
    cache_.decision_variables =
        Eigen::Map<Eigen::VectorXd>(const_cast<double*>(x), n);
  }
  // Update caches
  for (auto& binding : getCurrentProgram().f().all()) {
    cache_.objective += binding.get()->evaluate(cache_.decision_variables);
  }

  // Set objective to most recently cached value
  obj_value = objective_cache_;

  VLOG(10) << "f : " << obj_value;

  return true;
}

bool IpoptSolverInstance::eval_grad_f(Index n, const Number* x, bool new_x,
                                      Number* grad_f) {
  core::Profiler profiler("IpoptSolverInstance::eval_grad_f");
  VLOG(10) << "eval_grad_f()";

  if (new_x) {
    cache_.decision_variables =
        Eigen::Map<Eigen::VectorXd>(const_cast<double*>(x), n);
  }

  // Update caches
  for (auto& binding : getCurrentProgram().f().all()) {
    Eigen::RowVectorXd grd(binding.x().size());
    binding.get()->evaluate(cache_.decision_variables, grd);
    cache_.objective_gradient(getCurrentProgram().x().getIndices(binding.x())) =
        grd;
  }

  // TODO - See about mapping these
  std::copy_n(objective_gradient_cache_.data(), n, grad_f);
  VLOG(10) << "grad f : " << objective_gradient_cache_.transpose();
  return true;
}

bool IpoptSolverInstance::eval_g(Index n, const Number* x, bool new_x, Index m,
                                 Number* g) {
  core::Profiler profiler("IpoptSolverInstance::eval_g");
  VLOG(10) << "eval_g()";
  if (new_x) {
    cache_.decision_variables =
        Eigen::Map<Eigen::VectorXd>(const_cast<double*>(x), n);
  }

  // Update caches
  for (auto& binding : getCurrentProgram().g().all()) {
    Eigen::VectorXd g = binding.get()->evaluate(cache_.decision_variables);
    cache_.constraint_vector(getCurrentProgram().x().getIndices(binding.x())) =
        g;
  }
  VLOG(10) << "c : " << constraint_cache_.transpose();
  std::copy_n(constraint_cache_.data(), m, g);
  return true;
};

bool IpoptSolverInstance::eval_jac_g(Index n, const Number* x, bool new_x,
                                     Index m, Index nele_jac, Index* iRow,
                                     Index* jCol, Number* values) {
  core::Profiler profiler("IpoptSolverInstance::eval_jac_g");
  VLOG(10) << "eval_jac_g()";
  if (values == NULL) {
    // Return the sparsity of the constraint Jacobian
    const Eigen::SparseMatrix<double>& Jc = GetSparseConstraintJacobian();
    int cnt = 0;
    for (int k = 0; k < Jc.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(Jc, k); it; ++it) {
        if (cnt > nele_jac) {
          return false;
        }

        iRow[cnt] = it.row();
        jCol[cnt] = it.col();
        cnt++;
      }
    }

  } else {
    if (new_x) {
      cache_.decision_variables =
          Eigen::Map<Eigen::VectorXd>(const_cast<double*>(x), n);
    }
    // Update caches
    EvaluateConstraints(cache_.decision_variables, true, false);
    VLOG(10) << "x : " << cache_.decision_variables.transpose();
    VLOG(10) << "jac : " << constraint_jacobian_cache_;
    std::copy_n(constraint_jacobian_cache_.valuePtr(), nele_jac, values);
  }
  return true;
}

bool IpoptSolverInstance::eval_h(Index n, const Number* x, bool new_x,
                                 Number obj_factor, Index m,
                                 const Number* lambda, bool new_lambda,
                                 Index nele_hess, Index* iRow, Index* jCol,
                                 Number* values) {
  core::Profiler profiler("IpoptSolverInstance::eval_h");
  VLOG(10) << "eval_h()";
  if (values == NULL) {
    // Return the sparsity of the constraint Jacobian
    const Eigen::SparseMatrix<double>& H = GetSparseLagrangianHessian();
    int cnt = 0;
    for (int k = 0; k < H.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(H, k); it; ++it) {
        if (cnt > nele_hess) {
          return false;
        }

        iRow[cnt] = it.row();
        jCol[cnt] = it.col();
        cnt++;
      }
    }

  } else {
    if (new_x) {
      cache_.decision_variables =
          Eigen::Map<Eigen::VectorXd>(const_cast<double*>(x), n);
    }
    if (new_lambda) {
      dual_variable_cache_ =
          Eigen::Map<Eigen::VectorXd>(const_cast<double*>(lambda), m);
    }
    // Reset cache for hessian
    lagrangian_hes_cache_ *= 0.0;
    // Update caches
    EvaluateCosts(cache_.decision_variables, false, true);
    EvaluateConstraints(cache_.decision_variables, false, true);
    VLOG(10) << "hes\n" << lagrangian_hes_cache_;
    std::copy_n(lagrangian_hes_cache_.valuePtr(), nele_hess, values);
  }

  return true;
}

bool IpoptSolverInstance::get_bounds_info(Index n, Number* x_l, Number* x_u,
                                          Index m, Number* g_l, Number* g_u) {
  VLOG(10) << "get_bounds_info()";
  // Convert any bounding box constraints
  for (auto& binding : getCurrentProgram().GetBoundingBoxConstraintBindings()) {
    const sym::VariableVector& v = binding.x(0);
    for (int i = 0; i < v.size(); ++i) {
      getCurrentProgram().setDecisionVariableBounds(
          v[i], binding.Get().lowerBound()[i], binding.Get().upperBound()[i]);
    }
  }

  getCurrentProgram().updateDecisionVariableBoundVectors();
  getCurrentProgram().UpdateConstraintBoundVectors();
  VLOG(10) << getCurrentProgram().decisionVariableupperBounds();
  VLOG(10) << getCurrentProgram().decisionVariablelowerBounds();
  VLOG(10) << getCurrentProgram().ConstraintupperBounds();
  VLOG(10) << getCurrentProgram().ConstraintlowerBounds();
  // Decision variable bounds
  std::copy_n(getCurrentProgram().decisionVariableupperBounds().data(), n, x_u);
  std::copy_n(getCurrentProgram().decisionVariablelowerBounds().data(), n, x_l);
  // Constraint bounds
  std::copy_n(getCurrentProgram().ConstraintupperBounds().data(), m, g_u);
  std::copy_n(getCurrentProgram().ConstraintlowerBounds().data(), m, g_l);

  return true;
}

bool IpoptSolverInstance::get_starting_point(Index n, bool init_x, Number* x,
                                             bool init_z, Number* z_L,
                                             Number* z_U, Index m,
                                             bool init_lambda, Number* lambda) {
  VLOG(10) << "get_starting_point()";
  if (init_x) {
    getCurrentProgram().UpdateInitialValueVector();
    VLOG(10) << getCurrentProgram().decisionVariableInitialValues();
    std::copy_n(getCurrentProgram().decisionVariableInitialValues().data(), n,
                x);
  }

  return true;
}

void IpoptSolverInstance::finalize_solution(
    SolverReturn status, Index n, const Number* x, const Number* z_L,
    const Number* z_U, Index m, const Number* g, const Number* lambda,
    Number obj_value, const IpoptData* ip_data,
    IpoptCalculatedQuantities* ip_cq) {
  VLOG(10) << "finalize_solution()";
  for (Index i = 0; i < n; ++i) {
    VLOG(10) << x[i];
  }
}

int IpoptSolver::solve() {
  // Create a new instance of your nlp
  //  (use a SmartPtr, not raw)
  Ipopt::SmartPtr<Ipopt::TNLP> nlp = new IpoptSolverInstance(program_);

  Ipopt::SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

  app->Options()->SetNumericValue("tol", 1e-4);
  app->Options()->SetStringValue("mu_strategy", "adaptive");

  // Initialize the IpoptApplication and process the options
  Ipopt::ApplicationReturnStatus status;
  status = app->Initialize();
  if (status != Ipopt::Solve_Succeeded) {
    std::cout << std::endl
              << std::endl
              << "*** Error during initialization!" << std::endl;
    return (int)status;
  }

  // Ask Ipopt to solve the problem
  status = app->OptimizeTNLP(nlp);

  if (status == Solve_Succeeded) {
    std::cout << "*** The problem solved!" << std::endl;
  } else {
    std::cout << "*** The problem FAILED!" << std::endl;
  }

  return (int)status;
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion
