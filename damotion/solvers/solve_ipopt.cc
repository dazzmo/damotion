// #include "damotion/solvers/solve_ipopt.h"

// namespace damotion {
// namespace optimisation {

// using namespace Ipopt;
// using Ipopt::Index;
// using Ipopt::Number;

// IpoptSolverInstance::IpoptSolverInstance(Program& prog)
//     : prog_(prog), Ipopt::TNLP(), SolverBase(prog) {}

// bool IpoptSolverInstance::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
//                                        Index& nnz_h_lag,
//                                        IndexStyleEnum& index_style) {
//     n = prog_.NumberOfDecisionVariables();
//     m = prog_.NumberOfConstraints();
//     nnz_jac_g = prog_.ConstraintJacobian().nonZeros();
//     nnz_h_lag = prog_.LagrangianHessian().nonZeros();

//     index_style = TNLP::C_STYLE;
// }

// bool IpoptSolverInstance::eval_f(Index n, const Number* x, bool new_x,
//                                  Number& obj_value) {
//     if (new_x) {
//         decision_variable_cache_ =
//             Eigen::Map<VectorXd>(const_cast<double*>(x), n);
//     }
//     // Update caches
//     EvaluateCosts(n, decision_variable_cache_, false);
//     // Set objective to most recently cached value
//     obj_value = objective_cache_;
//     return true;
// }

// bool IpoptSolverInstance::eval_grad_f(Index n, const Number* x, bool new_x,
//                                       Number* grad_f) {
//     if (new_x) {
//         decision_variable_cache_ =
//             Eigen::Map<VectorXd>(const_cast<double*>(x), n);
//     }
//     // Update caches
//     EvaluateCosts(n, decision_variable_cache_, true);
//     std::copy_n(objective_gradient_cache_.data(), n, grad_f);
//     return true;
// }

// bool IpoptSolverInstance::eval_g(Index n, const Number* x, bool new_x, Index
// m,
//                                  Number* g) {
//     if (new_x) {
//         decision_variable_cache_ =
//             Eigen::Map<VectorXd>(const_cast<double*>(x), n);
//     }
//     // Update caches
//     EvaluateConstraints(n, decision_variable_cache_, false);
//     std::copy_n(constraint_cache_.data(), m, g);
//     return true;
// };

// bool IpoptSolverInstance::eval_jac_g(Index n, const Number* x, bool new_x,
//                                      Index m, Index nele_jac, Index* iRow,
//                                      Index* jCol, Number* values) {
//     if (values == NULL) {
//         // Return the sparsity of the constraint Jacobian
//         const Eigen::SparseMatrix<double>& Jc = prog_.ConstraintJacobian();
//         int cnt = 0;
//         for (int k = 0; k < Jc.outerSize(); ++k) {
//             for (Eigen::SparseMatrix<double>::InnerIterator it(Jc, k); it;
//                  ++it) {
//                 if (cnt > nele_jac) {
//                     return false;
//                 }

//                 iRow[cnt] = it.row();
//                 jCol[cnt] = it.col();
//                 cnt++;
//             }
//         }

//     } else {
//         if (new_x) {
//             decision_variable_cache_ =
//                 Eigen::Map<VectorXd>(const_cast<double*>(x), n);
//         }
//         // Update caches
//         EvaluateConstraints(n, decision_variable_cache_, true);
//         std::copy_n(constraint_jacobian_cache_.data(), nele_jac, values);
//     }
//     return true;
// }

// bool IpoptSolverInstance::eval_h(Index n, const Number* x, bool new_x,
//                                  Number obj_factor, Index m,
//                                  const Number* lambda, bool new_lambda,
//                                  Index nele_hess, Index* iRow, Index* jCol,
//                                  Number* values) {
//     if (values == NULL) {
//         // Return the sparsity of the constraint Jacobian
//         const Eigen::SparseMatrix<double>& H = prog_.LagrangianHessian();
//         int cnt = 0;
//         for (int k = 0; k < H.outerSize(); ++k) {
//             for (Eigen::SparseMatrix<double>::InnerIterator it(H, k); it;
//                  ++it) {
//                 if (cnt > nele_hess) {
//                     return false;
//                 }

//                 iRow[cnt] = it.row();
//                 jCol[cnt] = it.col();
//                 cnt++;
//             }
//         }

//     } else {
//         if (new_x) {
//             decision_variable_cache_ =
//                 Eigen::Map<VectorXd>(const_cast<double*>(x), n);
//         }
//         if (new_lambda) {
//             lambda_cache_ =
//                 Eigen::Map<VectorXd>(const_cast<double*>(lambda), m);
//         }
//         // Reset cache for hessian
//         lagrangian_hes_cache_.setZero();
//         // Update caches
//         EvaluateCostHessians(n, decision_variable_cache_, obj_factor);
//         EvaluateConstraintHessians(n, decision_variable_cache_, m,
//                                    lambda_cache_);
//         std::copy_n(lagrangian_hes_cache_.valuePtr(), nele_hess, values);
//     }

//     return true;
// }

// bool IpoptSolverInstance::get_bounds_info(Index n, Number* x_l, Number* x_u,
//                                           Index m, Number* g_l, Number* g_u)
//                                           {
//     // Decision variable bounds
//     std::copy_n(prog_.DecisionVariablesUpperBound().data(), n, x_u);
//     std::copy_n(prog_.DecisionVariablesLowerBound().data(), n, x_l);
//     // Constraint bounds
//     std::copy_n(prog_.ConstraintsUpperBound().data(), m, g_u);
//     std::copy_n(prog_.ConstraintsLowerBound().data(), m, g_l);

//     return true;
// }

// bool IpoptSolverInstance::get_starting_point(Index n, bool init_x, Number* x,
//                                              bool init_z, Number* z_L,
//                                              Number* z_U, Index m,
//                                              bool init_lambda, Number*
//                                              lambda) {
//     if (init_x) {
//         std::copy_n(prog_.DecisionVariablesInitialPoint().data(), n, x);
//     }
// }

// void IpoptSolverInstance::finalize_solution(
//     SolverReturn status, Index n, const Number* x, const Number* z_L,
//     const Number* z_U, Index m, const Number* g, const Number* lambda,
//     Number obj_value, const IpoptData* ip_data,
//     IpoptCalculatedQuantities* ip_cq) {}

// int IpoptSolver::solve() {
//     // Create a new instance of your nlp
//     //  (use a SmartPtr, not raw)
//     Ipopt::SmartPtr<Ipopt::TNLP> nlp = new IpoptSolverInstance(prog_);

//     // Create a new instance of IpoptApplication
//     //  (use a SmartPtr, not raw)
//     // We are using the factory, since this allows us to compile this
//     // example with an Ipopt Windows DLL
//     SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

//     // Change some options
//     // Note: The following choices are only examples, they might not be
//     //       suitable for your optimization problem.
//     app->Options()->SetNumericValue("tol", 3.82e-6);
//     app->Options()->SetStringValue("mu_strategy", "adaptive");
//     // Initialize the IpoptApplication and process the options
//     ApplicationReturnStatus status;
//     status = app->Initialize();
//     if (status != Solve_Succeeded) {
//         std::cout << std::endl
//                   << std::endl
//                   << "*** Error during initialization!" << std::endl;
//         return (int)status;
//     }

//     // Ask Ipopt to solve the problem
//     status = app->OptimizeTNLP(nlp);

//     if (status == Solve_Succeeded) {
//         std::cout << std::endl
//                   << std::endl
//                   << "*** The problem solved!" << std::endl;
//     } else {
//         std::cout << std::endl
//                   << std::endl
//                   << "*** The problem FAILED!" << std::endl;
//     }

//     // As the SmartPtrs go out of scope, the reference count
//     // will be decremented and the objects will automatically
//     // be deleted.

//     return (int)status;
// }

// }  // namespace optimisation
// }  // namespace damotion
