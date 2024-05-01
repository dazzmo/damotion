#ifndef SYMBOLIC_PARAMETER_H
#define SYMBOLIC_PARAMETER_H

#include <Eigen/Core>
#include <vector>

namespace damotion {
namespace symbolic {

typedef std::vector<Eigen::Ref<const Eigen::MatrixXd>> ParameterRefVector;

}
}  // namespace damotion

#endif /* SYMBOLIC_PARAMETER_H */
