#ifndef SOLVERS_BASE_H
#define SOLVERS_BASE_H

#include <unordered_map>

#include "damotion/optimisation/program.h"

namespace damotion {
namespace optimisation {
namespace solvers {

class SolverBase {
 public:
  class Context {
   public:
    using VectorType = Eigen::VectorXd;
    using MatrixType = Eigen::MatrixXd;

    VectorType primal;
    VectorType dual;
    double objective;
    VectorType objective_gradient;
    VectorType constraint_vector;

    Context() = default;
    ~Context() = default;

    Context(const std::size_t& nx, const std::size_t& ng) {
      // Initialise vectors
      primal = VectorType::Zero(nx);
      dual = VectorType::Zero(ng);
      objective = 0.0;
      objective_gradient = VectorType::Zero(nx);
      constraint_vector = VectorType::Zero(ng);
    }
  };

  SolverBase(MathematicalProgram& program) : program_(program) {}

  ~SolverBase() {}

  MathematicalProgram& getCurrentProgram() { return program_; }

 protected:
 private:
  bool is_solved_ = false;
  // Reference to current program in solver
  MathematicalProgram& program_;
};

enum class Operation { SET = 0, ADD };

/**
 * @brief Update the sparse matrix with a block, where the rows and entries of
 * the block are indexed in the sparse matrix using the provided index vectors.
 *
 * @param M
 * @param block
 * @param row_indices
 * @param col_indices
 */
void updateSparseMatrix(Eigen::SparseMatrix<double>& M,
                        const Eigen::MatrixXd& block,
                        const std::vector<Eigen::Index>& row_indices,
                        const std::vector<Eigen::Index>& col_indices,
                        const Operation& operation = Operation::SET);

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_BASE_H */
