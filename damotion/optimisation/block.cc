#include "damotion/optimisation/block.h"

namespace damotion {
namespace optimisation {

BlockMatrixFunction::BlockMatrixFunction(const int &rows, const int &cols,
                                         const Type &type)
    : type_(type), nc_(0) {
  // TODO - Make the flag for sparse or dense
  mat_.resize(rows, cols);
  mat_idx_.resize(rows, cols);
}

void BlockMatrixFunction::AddBinding(const BindingBase &binding,
                                     const GenericEigenMatrix &data,
                                     Program &program) {
  // Get indices of all variables within the program
  const sym::VariableVector &v = binding.GetConcatenatedVariableVector();
  std::vector<int> indices = program.getDecisionVariableIndices(v);

  // Add binding's data non-zero entry indices to a vector look-up
  binding_idx_[binding.id()].resize(data.nnz());

  int cnt = 0, i_idx = 0, j_idx = 0;
  for (int k = 0; k < data.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(data, k); it; ++it) {
      VLOG(10) << "it.row " << it.row() << ", it.col " << it.col();
      // Get index of the entry
      if (type_ == Type::kHessian || type_ == Type::kQuadratic) {
        // Only add the lower-triangle components
        VLOG(10) << "Hessian";
        i_idx = indices[it.row()];
        j_idx = indices[it.col()];
      } else if (type_ == Type::kJacobian) {
        // Add new constraint bindings constraint-by-constraint
        VLOG(10) << "Jacobian";
        VLOG(10) << "nc " << nc_;
        i_idx = nc_ + it.row();
        j_idx = indices[it.col()];
      } else if (type_ == Type::kGradient || type_ == Type::kLinear) {
        i_idx = 0;
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
        mat_idx_.coeffRef(i_idx, j_idx) = std::make_unique<EntryInfo>();
        // TODO - Make these modifyable values
        mat_idx_.coeffRef(i_idx, j_idx)->bindings.reserve(10);
        mat_idx_.coeffRef(i_idx, j_idx)->data_idx.reserve(100);
        mat_idx_.coeffRef(i_idx, j_idx)->bindings = {binding.id()};
        mat_idx_.coeffRef(i_idx, j_idx)->data_idx = {cnt};
      }
      // Set element in the full Jacobian to the information for
      // the Jacobian block provided
      VLOG(10) << "Adding element (" << i_idx << ", " << j_idx << ")";
      // Increase data array counter
      cnt++;
      // Increase constraint counter
      if (type_ == Type::kJacobian) nc_ += binding.Get().dim();
    }
  }
}

int BlockMatrixFunction::GenerateMatrix() {
  // Compress matrices
  mat_.makeCompressed();
  mat_idx_.makeCompressed();
  VLOG(10) << mat_;

  // Iterate through all non-zeros of the matrix and add to binding look-ups
  for (int i = 0; i < mat_idx_.nonZeros(); ++i) {
    const EntryInfo &data = *mat_idx_.valuePtr()[i];
    // For each binding, add to their data entries
    for (size_t j = 0; j < data->bindings.size(); ++j) {
      binding_idx_[data->bindings[j]][data->data_idx[j]] = i;
    }
  }

  // Delete the generator matrix memory
  mat_idx_.setZero();
  mat_idx_.data().squeeze();

  return 0;
}

int BlockMatrixFunction::Update(const BindingBase &binding,
                                const GenericEigenMatrix &data) {
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

}  // namespace optimisation
}  // namespace damotion
