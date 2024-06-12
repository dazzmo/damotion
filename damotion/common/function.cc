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

void Function::SetInputs(
    const std::vector<int> &indices,
    const std::vector<Eigen::Ref<const Eigen::VectorXd>> &input, bool check) {
  for (size_t i = 0; i < indices.size(); ++i) {
    SetInput(indices[i], input[i], check);
  }
}

}  // namespace common
}  // namespace damotion
