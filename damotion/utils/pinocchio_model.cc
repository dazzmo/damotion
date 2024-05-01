#include "damotion/utils/pinocchio_model.h"

namespace damotion {
namespace utils {
namespace casadi {

PinocchioModelWrapper &PinocchioModelWrapper::operator=(
    pinocchio::Model model) {
  // Cast model to type
  model_ = model.cast<::casadi::Matrix<AD>>();
  // Create data for model
  data_ = pinocchio::DataTpl<::casadi::Matrix<AD>>(model_);

  return *this;
}

::casadi::Function PinocchioModelWrapper::aba() {
  // Compute expression for aba
  ::casadi::Matrix<AD> q = ::casadi::Matrix<AD>::sym("q", model_.nq),
                       v = ::casadi::Matrix<AD>::sym("v", model_.nv),
                       tau = ::casadi::Matrix<AD>::sym("tau", model_.nv), a;
  // Convert to eigen expressions
  Eigen::VectorX<::casadi::Matrix<AD>> qe, ve, taue;
  toEigen(q, qe);
  toEigen(v, ve);
  toEigen(tau, taue);

  Eigen::VectorX<::casadi::Matrix<AD>> ae =
      pinocchio::aba<::casadi::Matrix<AD>>(model_, data_, qe, ve, taue);

  // Create AD equivalent
  toCasadi(ae, a);

  // Create function for ABA
  return ::casadi::Function(model_.name + "_aba", {q, v, tau}, {a},
                            {"q", "v", "u"}, {"a"});
}

::casadi::Function PinocchioModelWrapper::rnea() {
  // Compute expression for aba
  ::casadi::Matrix<AD> q = ::casadi::Matrix<AD>::sym("q", model_.nq),
                       v = ::casadi::Matrix<AD>::sym("v", model_.nv),
                       a = ::casadi::Matrix<AD>::sym("a", model_.nv), u;
  // Convert to eigen expressions
  Eigen::VectorX<::casadi::Matrix<AD>> qe, ve, ae;
  toEigen(q, qe);
  toEigen(v, ve);
  toEigen(a, ae);

  // Convert to SX
  Eigen::VectorX<::casadi::Matrix<AD>> ue =
      pinocchio::rnea<::casadi::Matrix<AD>>(model_, data_, qe, ve, ae);

  toCasadi(ue, u);

  // Create function for RNEA
  return ::casadi::Function(model_.name + "_rnea", {q, v, a}, {u},
                            {"q", "v", "a"}, {"u"});
};

}  // namespace casadi
}  // namespace utils
}  // namespace damotion
