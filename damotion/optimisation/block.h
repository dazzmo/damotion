/**
 * @file block.h
 * @author Damian Abood (damian.abood@sydney.edu.au)
 * @brief Block matrix class to handle assemblage of large matrices such as
 * Jacobians and Hessians from bindings for use in numerical optimisation.
 * @version 0.1
 * @date 2024-05-15
 *
 */
#ifndef OPTIMISATION_BLOCK_H
#define OPTIMISATION_BLOCK_H

#include <Eigen/Sparse>
#include <unordered_map>
#include <vector>

#include "damotion/common/eigen.h"
#include "damotion/optimisation/binding.h"
#include "damotion/optimisation/fwd.h"
#include "damotion/optimisation/program.h"

namespace damotion {
namespace optimisation {

class BlockMatrixFunction {
 public:
  /**
   * @brief Block matrix types that can be represented under the
   * BlockMatrixFunction class
   *
   */
  enum class Type { kJacobian, kHessian, kGradient, kQuadratic, kLinear };

  /**
   * @brief Construct a new Block Matrix Function object
   *
   * @param rows
   * @param cols
   * @param type
   * @param sparse
   */
  BlockMatrixFunction(const int &rows, const int &cols, const Type &type);

  void AddBinding(const BindingBase &binding, const GenericEigenMatrix &data,
                  Program &program);

  /**
   * @brief Generates the complete matrix and its sparsity pattern
   *
   * @return int
   */
  int GenerateMatrix();

  /**
   * @brief The current matrix
   *
   * @return const GenericEigenMatrix&
   */
  const GenericEigenMatrix &GetMatrix() const { return mat_; }

  /**
   * @brief Update the matrix with the current values given by the binding
   *
   * @tparam T
   * @param binding
   * @param data Matrix data to update the block matrix with
   */
  int Update(const BindingBase &binding, const GenericEigenMatrix &data);

 private:
  struct EntryInfo {
    using UniquePtr = std::unique_ptr<EntryInfo>;
    using SharedPtr = std::shared_ptr<EntryInfo>;
    // Bindings that have a value at this entry
    std::vector<BindingBase::Id> bindings;
    // Index of the data vector for each binding entry
    std::vector<int> data_idx;
  };

  Type type_;
  int nc_;

  GenericEigenMatrix mat_;

  Eigen::SparseMatrix<EntryInfo::UniquePtr> mat_idx_;
  std::unordered_map<BindingBase::Id, std::vector<int>> binding_idx_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* OPTIMISATION_BLOCK_H */
