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
    std::cout << "x : " << x << '\n';
    std::cout << "jac : " << jac << '\n';

    // Place into the jacobian
    res.middleRows(cnt, binding.get()->size())(Eigen::all, indices) = jac;
    cnt += binding.get()->size();
  }

  std::cout << "res : " << res << '\n';

  return res;
}

Eigen::MatrixXd constraintHessian(const Eigen::VectorXd &x,
                                  const Eigen::VectorXd &lam,
                                  const ConstraintVector &g,
                                  const symbolic::VariableVector &v) {
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(v.size(), v.size());
  // Loop through all constraints
  std::size_t cnt = 0;
  for (const auto &binding : g.all()) {
    // Evaluate the Hessian
    Eigen::MatrixXd hes(binding.x().size(), binding.x().size());
    auto indices = v.getIndices(binding.x());
    binding.get()->hessian(x(indices),
                           lam.middleRows(cnt, binding.get()->size()), hes);

    // Place into the hessian
    res(indices, indices) += hes;
    cnt += binding.get()->size();
  }

  return res;
}

Eigen::MatrixXd objectiveHessian(const Eigen::VectorXd &x,
                                 const ObjectiveFunction &f,
                                 const symbolic::VariableVector &v) {
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(v.size(), v.size());
  // Loop through all objectives
  for (const auto &binding : f.all()) {
    // Evaluate the Jacobian
    Eigen::MatrixXd hes(binding.x().size(), binding.x().size());
    auto indices = v.getIndices(binding.x());
    // Evaluate the hessian
    binding.get()->hessian(x(indices), 1.0, hes);
    // Add to the hessian
    res(indices, indices) += hes;
  }

  return res;
}

}  // namespace optimisation
}  // namespace damotion
