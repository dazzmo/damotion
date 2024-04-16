#include "common/function.h"

namespace damotion {
namespace common {

Function::Function(const Function& other) { *this = other; }

Function& Function::operator=(const Function& other) { return *this; }

Function::~Function() {}

void Function::setSparseOutput(int i) {
    // Set output flag to true and remove dense matrix data
    is_out_sparse_[i] = true;
    out_[i].resize(0, 0);
}

const Eigen::Ref<const Eigen::MatrixXd> Function::getOutput(int i) {
    // Return the dense output i
    LOG(INFO) << "i: " << i << std::endl;
    LOG(INFO) << "Size: " << out_[i].rows() << " x " << out_[i].cols() << std::endl;
    LOG(INFO) << "Value: " << out_[i] << std::endl;
    return out_[i];
}

const Eigen::Ref<const Eigen::SparseMatrix<double>> Function::getOutputSparse(
    int i) {
    if (is_out_sparse_[i] != true) {
        // ! Throw warning and create sparse output
    }
    // Return the sparse output i
    return out_sparse_[i];
}

}  // namespace common
}  // namespace damotion