#include "common/function.h"

namespace damotion {
namespace common {

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