#include "damotion/utils/casadi.h"

namespace damotion {
namespace utils {
namespace casadi {

::casadi::StringVector CreateInputNames(::casadi::Function &f) {
  ::casadi::StringVector in;
  for (int i = 0; i < f.n_in(); ++i) {
    // Add inputs to constraint
    in.push_back(f.name_in(i));
  }
  return in;
}

::casadi::Function CreateGradientFunction(
    const std::string &name, const ::casadi::SX &f,
    const ::casadi::SXVector &in, const ::casadi::StringVector &inames,
    const ::casadi::SXVector &x, const ::casadi::StringVector &xnames) {
  assert(in.size() == inames.size() &&
         "Number of inputs and input names don't match!");
  assert(x.size() == xnames.size() &&
         "Number of independent variables and names don't match!");
  // Vector of gradients and output names
  ::casadi::SXVector g;
  ::casadi::StringVector gnames;
  // For each entry, compute the gradient
  for (int i = 0; i < x.size(); ++i) {
    // Compute gradients of cost with respect to variable set x
    g.push_back(gradient(f, x[i]));
    gnames.push_back("grad_" + xnames[i]);
  }
  // Create function
  return ::casadi::Function(name + "_grad", in, g, inames, gnames);
}

::casadi::Function CreateJacobianFunction(
    const std::string &name, const ::casadi::SX &f,
    const ::casadi::SXVector &in, const ::casadi::StringVector &inames,
    const ::casadi::SXVector &x, const ::casadi::StringVector &xnames) {
  assert(in.size() == inames.size() &&
         "Number of inputs and input names don't match!");
  assert(x.size() == xnames.size() &&
         "Number of independent variables and names don't match!");
  // Vector of gradients and output names
  ::casadi::SXVector jac;
  ::casadi::StringVector jacnames;
  // For each entry, compute the gradient
  for (int i = 0; i < x.size(); ++i) {
    // Compute gradients of cost with respect to variable set x
    jac.push_back(jacobian(f, x[i]));
    jacnames.push_back("jac_" + xnames[i]);
  }
  // Create function
  return ::casadi::Function(name + "_jac", in, jac, inames, jacnames);
}

::casadi::Function CreateHessianFunction(
    const std::string &name, const ::casadi::SX &f,
    const ::casadi::SXVector &in, const ::casadi::StringVector &inames,
    const std::vector<std::pair<::casadi::SX, ::casadi::SX>> &xy,
    const std::vector<std::pair<std::string, std::string>> &xynames) {
  // Vector of gradients and output names
  ::casadi::SXVector h;
  ::casadi::StringVector hnames;
  // For each entry, compute the gradient
  for (int i = 0; i < xy.size(); ++i) {
    // Compute gradients of cost with respect to variable set x
    h.push_back(jacobian(gradient(f, xy[i].first), xy[i].second));
    hnames.push_back("hes_" + xynames[i].first + "_" + xynames[i].second);
  }

  // Create function
  return ::casadi::Function(name + "_hes", in, h, inames, hnames);
}

}  // namespace casadi
}  // namespace utils
}  // namespace damotion
