#include "damotion/optimisation/program.h"

namespace damotion {
namespace optimisation {

Eigen::MatrixXd constraintJacobian(const Eigen::VectorXd &x,
                                   const ConstraintVector &g,
                                   const symbolic::VariableVector &v) {
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(g.size(), v.size());
  // Loop through all constraints
  std::size_t cnt = 0;
  for (const auto &binding : g.all()) {
    // Evaluate the Jacobian
    Eigen::MatrixXd jac(binding.get()->size(), binding.x().size());
    auto indices = v.getIndices(binding.x());
    binding.get()->evaluate(x(indices), jac);

    // Place into the jacobian
    res.middleRows(cnt, binding.get()->size())(Eigen::all, indices) = jac;
    cnt += binding.get()->size();
  }

  return res;
}

}  // namespace optimisation
}  // namespace damotion
