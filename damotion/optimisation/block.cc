#include "damotion/optimisation/block.h"

namespace damotion {
namespace optimisation {

void BlockMatrixFunction::AddBinding(
    const Binding<common::Expression<Eigen::MatrixXd>> &binding,
    const int &constraint_idx) {
  const sym::VariableVector &v = binding.GetConcatenatedVariableVector();

  const DenseMatrix &M = GetBindingMatrix(binding, GetType());

  assert(M.rows() <= rows() && M.cols() <= cols() &&
         "Binding matrix is too large for block matrix");

  std::vector<bool> continuous;
  for (int i = 0; i < binding.nx(); ++i) {
    continuous.push_back(
        x_manager_.IsContinuousInDecisionVariableVector(binding.x(i)));
  }
  // Indicate if block insertion can happen
  binding_block_insert_[binding.id()] = continuous;
}

int BlockMatrixFunction::GenerateMatrix() {
  mat_.resize(rows(), cols());
  mat_.setZero();
  return 0;
}

int BlockMatrixFunction::Update(
    const Binding<common::Expression<Eigen::MatrixXd>> &binding,
    const int &constraint_idx) {
  // Update the matrix with the current values for the binding
  if (binding_block_insert_.find(binding.id()) == binding_block_insert_.end()) {
    // If binding does not exist, return non-zero value
    // TODO make error message
    return 1;
  }
  // Get matrix to update
  const DenseMatrix &M = GetBindingMatrix(binding, this->GetType());
  std::vector<bool> &continuous = binding_block_insert_[binding.id()];

  // Jacobian-based insertion
  if (this->GetType() == Type::kJacobian ||
      this->GetType() == Type::kGradient || this->GetType() == Type::kLinear) {
    // Add each entry by column
    for (int i = 0; i < binding.nx(); ++i) {
      const sym::VariableVector &xi = binding.x(i);
      Eigen::Ref<Eigen::MatrixXd> J = mat_.middleRows(constraint_idx, M.rows());
      if (continuous[i]) {
        J.middleCols(x_manager_.GetDecisionVariableIndex(xi[0]), xi.size()) +=
            M;
      } else {
        // For each variable, update the location in the Jacobian
        for (int i = 0; i < xi.size(); ++i) {
          int idx = x_manager_.GetDecisionVariableIndex(xi[i]);
          J.col(idx) += M.col(i);
        }
      }
    }

    // Hessian-based insertion
    if (this->GetType() == Type::kHessian ||
        this->GetType() == Type::kQuadratic) {
      // Move through the matrix
      int idx_x = 0, idx_y = 0;

      for (int ii = 0; ii < binding.nx(); ++ii) {
        const sym::VariableVector &xi = binding.x(ii);
        for (int jj = ii; jj < binding.nx(); ++jj) {
          const sym::VariableVector &xj = binding.x(jj);

          Eigen::Ref<const DenseMatrix> Hij =
              M.block(idx_x, idx_y, xi.size(), xj.size());

          // For each variable combination
          if (continuous[ii] && continuous[jj]) {
            int i_idx = x_manager_.GetDecisionVariableIndex(xi[0]);
            int j_idx = x_manager_.GetDecisionVariableIndex(xj[0]);
            // Create lower triangular Hessian
            if (i_idx > j_idx) {
              mat_.block(i_idx, j_idx, xi.size(), xj.size()) += Hij;
            } else {
              mat_.block(j_idx, i_idx, xj.size(), xi.size()) += Hij.transpose();
            }

          } else {
            // For each variable pair, populate the Hessian
            for (int i = 0; i < xi.size(); ++i) {
              int i_idx = x_manager_.GetDecisionVariableIndex(xi[i]);
              for (int j = 0; j < xj.size(); ++j) {
                int j_idx = x_manager_.GetDecisionVariableIndex(xj[j]);
                // Create lower triangular matrix
                if (i_idx > j_idx) {
                  mat_(i_idx, j_idx) += Hij(i, j);
                } else {
                  mat_(j_idx, i_idx) += Hij(i, j);
                }
              }
            }
          }
          idx_y += xj.size();
        }
        idx_x += xi.size();
      }
    }
  }

  // Return zero on success
  return 0;
}

void SparseBlockMatrixFunction::AddBinding(
    const Binding<common::Expression<SparseMatrix>> &binding,
    const int &constraint_idx) {
  // Create vector
  const sym::VariableVector &v = binding.GetConcatenatedVariableVector();
  const std::vector<int> &indices = x_manager_.GetDecisionVariableIndices(v);
  // Get sparse Jacobian
  const SparseMatrix &M = GetBindingMatrix(binding, GetType());

  VLOG(10) << "M\n" << M;

  int cnt = 0;

  // Add binding data vector to look-up
  binding_data_indices_[binding.id()].resize(M.nonZeros());

  int i_idx, j_idx;
  for (int k = 0; k < M.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
      VLOG(10) << "it.row " << it.row() << ", it.col " << it.col();
      // Get index of the entry
      if (type_ == Type::kHessian || type_ == Type::kQuadratic) {
        i_idx = indices[it.row()];
        j_idx = indices[it.col()];
      } else if (type_ == Type::kJacobian) {
        // Set row for the binding
        i_idx = constraint_idx + it.row();
        j_idx = indices[it.col()];
      } else if (type_ == Type::kLinear || type_ == Type::kGradient) {
        // Single column vector
        i_idx = indices[it.col()];
        j_idx = 0;
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

    // Increase constraint counter
    if (type_ == Type::kJacobian) nc_ += binding.Get().Dimension();
  }
}

int SparseBlockMatrixFunction::Update(
    const Binding<common::Expression<SparseMatrix>> &binding,
    const int &constraint_idx) {
  // Update the matrix with the current values for the binding
  if (binding_data_indices_.find(binding.id()) == binding_data_indices_.end()) {
    // If binding does not exist, return non-zero value
    // TODO make error message
    return 1;
  }
  // Get matrix to update
  const SparseMatrix &M = GetBindingMatrix(binding, GetType());
  // For each Jacobian, update the data within the constraint Jacobian
  std::vector<int> &indices = binding_data_indices_[binding.id()];
  for (int k = 0; k < M.nonZeros(); ++k) {
    mat_.valuePtr()[indices[k]] += M.valuePtr()[k];
  }

  return 0;
}

int SparseBlockMatrixFunction::GenerateMatrix() {
  // Compress matrices
  mat_.makeCompressed();
  VLOG(10) << mat_;
  mat_idx_.makeCompressed();

  // Iterate through all non-zeros of the matrix and add to binding look-ups
  for (int i = 0; i < mat_.nonZeros(); ++i) {
    EntryInfo::SharedPtr data = mat_idx_.valuePtr()[i];
    // For each binding, add to their data entries
    for (size_t j = 0; j < data->bindings.size(); ++j) {
      binding_data_indices_[data->bindings[j]][data->data_idx[j]] = i;
    }
  }

  // Delete the generator matrix

  return 0;
}
}  // namespace optimisation
}  // namespace damotion
