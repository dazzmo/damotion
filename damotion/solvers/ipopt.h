#ifndef SOLVERS_IPOPT_H
#define SOLVERS_IPOPT_H

#include <coin-or/IpIpoptApplication.hpp>
#include <coin-or/IpTNLP.hpp>

#include "damotion/common/logging.h"
#include "damotion/common/profiler.h"
#include "damotion/optimisation/program.h"
#include "damotion/solvers/sparse.h"

namespace damotion {
namespace optimisation {
namespace solvers {

using namespace Ipopt;

class IpoptSolverInstance : public Ipopt::TNLP, public SparseSolver {
 public:
  IpoptSolverInstance(SparseProgram& prog);

  ~IpoptSolverInstance() {}

  bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag,
                    IndexStyleEnum& index_style);

  bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number* g_l,
                       Number* g_u);

  bool get_starting_point(Index n, bool init_x, Number* x, bool init_z,
                          Number* z_L, Number* z_U, Index m, bool init_lambda,
                          Number* lambda);

  bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value);

  bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f);

  bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g);

  bool eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nele_jac,
                  Index* iRow, Index* jCol, Number* values);

  bool eval_h(Index n, const Number* x, bool new_x, Number obj_factor, Index m,
              const Number* lambda, bool new_lambda, Index nele_hess,
              Index* iRow, Index* jCol, Number* values);

  void finalize_solution(SolverReturn status, Index n, const Number* x,
                         const Number* z_L, const Number* z_U, Index m,
                         const Number* g, const Number* lambda,
                         Number obj_value, const IpoptData* ip_data,
                         IpoptCalculatedQuantities* ip_cq);

 private:
};

class IpoptSolver {
 public:
  IpoptSolver(SparseProgram& program) : program_(program) {}

  int solve();

 private:
  SparseProgram& program_;
};

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_IPOPT_H */
