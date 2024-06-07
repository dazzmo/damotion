#include "damotion/common/function.h"

namespace damotion {
namespace common {

void Function::SetInput(const int &i,
                        const Eigen::Ref<const Eigen::VectorXd> &input,
                        bool check) {
  assert(i < n_in_ && "Index out of bounds");
  if (check) {
    if (input.hasNaN() || !input.allFinite()) {
      std::ostringstream ss;
      ss << "Input " << i << " has invalid values:\n"
         << input.transpose().format(3);
      throw std::runtime_error(ss.str());
    }
  }
  in_[i] = input.data();
}

void Function::SetInput(
    const std::vector<int> &indices,
    const std::vector<Eigen::Ref<const Eigen::VectorXd>> &input, bool check) {
  for (size_t i = 0; i < indices.size(); ++i) {
    SetInput(indices[i], input[i], check);
  }
}

void Function::SetInput(const int &i, const double *input, bool check) {
  assert(i < n_in_ && "Index out of bounds");
  if (check) {
    if (input == NULL) {
      throw std::runtime_error("Input " + std::to_string(i) +
                               "is invalid memory");
    }
  }
  in_[i] = input;
}

void Function::SetInput(const std::vector<int> &indices,
                        const std::vector<const double *> input,
                        bool check = false) {
  for (size_t i = 0; i < indices.size(); ++i) {
    SetInput(indices[i], input[i], check);
  }
}

template <>
void CallbackFunction<double>::InitialiseOutput(const int i,
                                                const Sparsity &sparsity) {
  this->OutputVector()[i] = 0.0;
}

template <>
void CallbackFunction<Eigen::SparseMatrix<double>>::InitialiseOutput(
    const int i, const Sparsity &sparsity) {
  this->OutputVector()[i] =
      common::CreateSparseEigenMatrix(const_cast<Sparsity &>(sparsity));
}

}  // namespace common
}  // namespace damotion
