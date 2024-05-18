#ifndef BLOCK_DENSE_H
#define BLOCK_DENSE_H

namespace damotion {
namespace optimisation {

/**
 * @brief Block matrix function that assembles a block matrix from a series of
 * bindings for a given program.
 *
 */
class BlockMatrixFunction : public BlockMatrixFunctionBase {
 public:
  BlockMatrixFunction(const int &rows, const int &cols, const Type &type,
                      DecisionVariableManager &x_manager)
      : BlockMatrixFunctionBase(rows, cols, type, x_manager) {
    mat_.resize(rows, cols);
  }

  /**
   * @brief Add a binding to the block matrix, using the orderings of the
   * variables given by indices
   *
   * @param binding
   * @param constraint_idx Starting index of the binding constraints, if
   * updating a constraint vector or Jacobian.
   */
  void AddBinding(const BindingBase &binding, const MatrixType &block,
                  const int &constraint_idx = 0);

  /**
   * @brief Update the block matrix with the current values provided by the
   * binding.
   *
   * @param binding
   * @param constraint_idx Starting index of the binding constraints, if
   * updating a constraint vector or Jacobian.
   */
  int Update(const BindingBase &binding, const MatrixType &block,
             const int &constraint_idx = 0);

  /**
   * @brief Generate the block matrix with the provided bindings.
   *
   * @return int
   */
  int GenerateMatrix();

  /**
   * @brief Resets the values of the matrix to zero.
   *
   */
  void ResetValues() { mat_.setZero(); }

 private:
  // Sparse data handling
  struct EntryInfo {
    using SharedPtr = std::shared_ptr<EntryInfo>;
    // Bindings that have a value at this entry
    std::vector<BindingBase::Id> bindings;
    // Index of the data vector for each binding entry
    std::vector<int> data_idx;
  };
  Eigen::SparseMatrix<EntryInfo::SharedPtr> mat_idx_;

  std::unordered_map<BindingBase::Id, std::vector<int>> binding_data_indices_;

  MatrixType mat_;
  // For dense matrices, whether each variable set can be block-inserted
  std::unordered_map<BindingBase::Id, std::vector<bool>> binding_block_insert_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* BLOCK_DENSE_H */
