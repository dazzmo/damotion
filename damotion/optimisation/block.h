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

#include "damotion/optimisation/binding.h"
#include "damotion/optimisation/fwd.h"
#include "damotion/optimisation/program.h"

namespace damotion {
namespace optimisation {

class BlockMatrixFunction {
 public:
  enum class Type { kJacobian, kHessian, kGradient, kQuadratic, kLinear };

  BlockMatrixFunction(const int &rows, const int &cols,
                      const Type &type = Type::kJacobian)
      : type_(type), nc_(0) {
    mat_.resize(rows, cols);
    mat_idx_.resize(rows, cols);
  }

  template <typename T>
  void AddBinding(const Binding<T> &binding, DecisionVariableManager &manager) {
    const sym::VariableVector &v = binding.GetConcatenatedVariableVector();

    // std::vector<int> indices = manager.GetDecisionVariableIndices();

    // Get sparse Jacobian
    Eigen::SparseMatrix<double> M;
    if (type_ == Type::kHessian || type_ == Type::kQuadratic) {
      M = binding.Get().Hessian();
    } else if (type_ == Type::kJacobian) {
      M = binding.Get().Jacobian();
    }

    VLOG(10) << "M = \n";
    VLOG(10) << M;

    int cnt = 0;
    std::vector<int> indices = manager.GetDecisionVariableIndices(v);

    // Add binding data vector to look-up
    binding_idx_[binding.id()].resize(M.nonZeros());

    int i_idx, j_idx;
    for (int k = 0; k < M.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
        VLOG(10) << "it.row " << it.row() << ", it.col " << it.col();
        // Get index of the entry
        if (type_ == Type::kHessian) {
          VLOG(10) << "Hessian";
          i_idx = indices[it.row()];
          j_idx = indices[it.col()];
        } else if (type_ == Type::kJacobian) {
          VLOG(10) << "Jacobian";
          VLOG(10) << "nc " << nc_;
          i_idx =
              it.row() + nc_;  // TODO Offset this by the number of constraints
          j_idx = indices[it.col()];
        }

        VLOG(10) << "New element (" << i_idx << ", " << j_idx << ")";

        // Check if entry already exists
        if (mat_idx_.coeff(i_idx, j_idx) != nullptr) {
          VLOG(10) << "Existing entry";
          // Add binding entry to data
          mat_idx_.coeffRef(i_idx, j_idx)->bindings.push_back(binding.id());
          mat_idx_.coeffRef(i_idx, j_idx)->data_idx.push_back(cnt);
        } else {
          VLOG(10) << "New entry";
          // Add a new entry to the matrix
          mat_.coeffRef(i_idx, j_idx) = 0.0;
          mat_idx_.coeffRef(i_idx, j_idx) = std::make_shared<EntryInfo>();
          mat_idx_.coeffRef(i_idx, j_idx)->bindings = {binding.id()};
          mat_idx_.coeffRef(i_idx, j_idx)->data_idx = {cnt};
        }
        // Set element in the full Jacobian to the information for
        // the Jacobian block provided
        VLOG(10) << "Adding element (" << i_idx << ", " << j_idx << ")";
        // Increase data array counter
        cnt++;
      }
    }

    // Increase constraint counter
    if (type_ == Type::kJacobian) nc_ += binding.Get().Dimension();
  }

  int GenerateMatrix() {
    // Compress matrices
    mat_.makeCompressed();
    VLOG(10) << mat_;
    mat_idx_.makeCompressed();

    // Iterate through all non-zeros of the matrix and add to binding look-ups
    for (int i = 0; i < mat_.nonZeros(); ++i) {
      EntryInfo::SharedPtr data = mat_idx_.valuePtr()[i];
      // For each binding, add to their data entries
      for (size_t j = 0; j < data->bindings.size(); ++j) {
        binding_idx_[data->bindings[j]][data->data_idx[j]] = i;
      }
    }

    // Delete the generator matrix

    return 0;
  }

  /**
   * @brief The current matrix
   *
   * @return const Eigen::SparseMatrix<double>&
   */
  const Eigen::SparseMatrix<double> &GetMatrix() const { return mat_; }

  /**
   * @brief Update the matrix with the current values given by the binding
   *
   * @tparam T
   * @param binding
   */
  template <typename T>
  int Update(const Binding<T> &binding) {
    // Update the matrix with the current values for the binding
    if (binding_idx_.find(binding.id()) == binding_idx_.end()) {
      // If binding does not exist, return non-zero value
      // TODO make error message
      return 1;
    }
    // Get matrix to update
    const Eigen::SparseMatrix<double> &M = GetBindingMatrix(binding, type_);
    // For each Jacobian, update the data within the constraint Jacobian
    std::vector<int> &indices = binding_idx_[binding.id()];
    for (int k = 0; k < M.nonZeros(); ++k) {
      mat_.valuePtr()[indices[k]] += M.valuePtr()[k];
    }

    return 0;
  }

  template <typename T>
  const Eigen::SparseMatrix<double> &GetBindingMatrix(
      const Binding<T> &binding, const Type &type = Type::kHessian) {
    if (type == Type::kHessian) return binding.Get().Hessian();
    if (type == Type::kJacobian) return binding.Get().Jacobian();
    if (type == Type::kQuadratic) return binding.Get().A();
    if (type == Type::kLinear) return binding.Get().c();
    // Default
    return binding.Get().Hessian();
  }

 private:
  struct EntryInfo {
    using SharedPtr = std::shared_ptr<EntryInfo>;
    // Bindings that have a value at this entry
    std::vector<BindingBase::Id> bindings;
    // Index of the data vector for each binding entry
    std::vector<int> data_idx;
  };

  int nc_;

  Type type_;
  Eigen::SparseMatrix<double> mat_;
  Eigen::SparseMatrix<EntryInfo::SharedPtr> mat_idx_;
  std::unordered_map<BindingBase::Id, std::vector<int>> binding_idx_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* OPTIMISATION_BLOCK_H */
