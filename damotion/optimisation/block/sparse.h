#ifndef BLOCK_SPARSE_H
#define BLOCK_SPARSE_H

/**
 * @brief Block matrix function that assembles a block matrix from a series of
 * bindings for a given program.
 *
 */
class SparseBlockMatrixFunction : public BlockMatrixFunctionBase {
 public:
  SparseBlockMatrixFunction(const int &rows, const int &cols, const Type &type,
                            DecisionVariableManager &x_manager)
      : BlockMatrixFunctionBase(rows, cols, type, x_manager) {
    mat_.resize(rows, cols);
    mat_idx_.resize(rows, cols);
  }

  /**
   * @brief The current matrix
   *
   * @return const MatrixType&
   */
  const SparseMatrix &GetMatrix() const { return mat_; }

  /**
   * @brief Resets the values of the matrix to zero.
   *
   */
  void ResetValues() {
    for (int i = 0; i < mat_.nonZeros(); ++i) {
      mat_.valuePtr()[i] = 0.0;
    }
  }

  /**
   * @brief Add a binding to the block matrix, using the orderings of the
   * variables given by indices
   *
   * @param binding
   * @param indices
   */
  void AddBinding(const Binding<common::Expression<SparseMatrix>> &binding,
                  const int &constraint_idx = 0);

  /**
   * @brief Generate the block matrix given the current set of bindings.
   *
   * @return int
   */
  int GenerateMatrix();

  /**
   * @brief Update the matrix with the current values given by the binding
   *
   * @param binding
   * @param constraint_idx Starting index of the binding constraints, if
   * updating a constraint vector or Jacobian.
   */
  int Update(const Binding<common::Expression<SparseMatrix>> &binding,
             const int &constraint_idx = 0);

 private:
};

#endif /* BLOCK_SPARSE_H */
