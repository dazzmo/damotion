#include "damotion/optimisation/program.h"

namespace damotion {
namespace optimisation {

std::ostream &operator<<(std::ostream &os, const ConstraintVector &cv) {
  std::ostringstream oss;
  oss << "Constraint\tSize\tLower Bound\tUpper Bound\n";
  // Get all constraints
  for (const Binding<Constraint> &b : cv.all()) {
    oss << b.get()->name() << "\t[" << b.get()->size() << ",1]\n";
    for (size_t i = 0; i < b.get()->size(); ++i) {
      oss << b.get()->name() << "_" + std::to_string(i) << "\t\t"
          << b.get()->lb()[i] << "\t" << b.get()->ub()[i] << "\n";
    }
  }

  return os << oss.str();
}

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
    VLOG(10) << "x : " << x;
    VLOG(10) << "jac : " << jac;

    // Place into the jacobian
    res.middleRows(cnt, binding.get()->size())(Eigen::all, indices) = jac;
    cnt += binding.get()->size();
  }

  VLOG(10) << "res : " << res;

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

std::ostream &operator<<(std::ostream &os, const ObjectiveFunction &obj) {
  std::ostringstream oss;
  // Get all costs
  for (const Binding<Cost> &b : obj.all()) {
    oss << b.get()->name() << "\n";
  }

  return os << oss.str();
}

}  // namespace optimisation
}  // namespace damotion
