#ifndef SOLVERS_BASE_H
#define SOLVERS_BASE_H

#include <unordered_map>

#include "damotion/optimisation/program.h"

namespace damotion {
namespace optimisation {
namespace solvers {

class SolverBase {
 public:
  using VectorType = Eigen::VectorXd;
  using MatrixType = Eigen::MatrixXd;

  class CacheData {
   public:
    using VectorType = Eigen::VectorXd;
    using MatrixType = Eigen::MatrixXd;

    VectorType decision_variable;
    VectorType primal_solution_x;
    VectorType primal_solution_g;
    double objective_cache;
    VectorType objective_gradient;
    VectorType constraint_vector;
    VectorType dual_vector;

    CacheData(const std::size_t& nx, const std::size_t& ng) {
      // Initialise vectors
      decision_variable = VectorType::Zero(nx);
      primal_solution_x = VectorType::Zero(nx);
      primal_solution_g = VectorType::Zero(ng);
      objective_cache = 0.0;
      objective_gradient = VectorType::Zero(nx);
      constraint_vector = VectorType::Zero(ng);
      dual_vector = VectorType::Zero(ng);
    }
  };

  class Derivative {
   public:
    VectorType objective_gradient;
    MatrixType constraint_jacobian;
  };

  SolverBase(MathematicalProgram& program) : program_(program) {
    int nx = program.x().size();
    int ng = program.g().size();

    CacheData cache_(nx, ng);
  }

  ~SolverBase() {}

 protected:
  CacheData cache_;

 private:
  bool is_solved_ = false;
  // Reference to current program in solver
  MathematicalProgram& program_;
};

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_BASE_H */
