#include "damotion/common/math/log.h"

namespace damotion {
namespace math {

template <>
Eigen::Matrix<::casadi::SX, 3, -1> log3(
    const Eigen::Matrix<::casadi::SX, 3, 3> &R, ::casadi::SX &theta) {
  Eigen::Vector<::casadi::SX, 3> res;

  // Upper and lower bounds for the trace of R
  ::casadi::SX ub = ::casadi::SX(3);
  ::casadi::SX lb = ::casadi::SX(-1);

  // Compute trace of matrix
  ::casadi::SX tr = R.trace();

  // Create conditions for checking bounds
  ::casadi::SX trace_greater_than_ub = tr >= ub;
  ::casadi::SX trace_less_than_lb = tr <= lb;

  // Determine trace
  ::casadi::SX cond_trace = if_else(trace_less_than_lb, lb, R.trace());
  ::casadi::SX trace = if_else(trace_greater_than_ub, ub, cond_trace);

  // Determine theta based on given conditions
  ::casadi::SX cond_theta =
      if_else(trace_less_than_lb, M_PI, acos((tr - 1) / 2));
  ::casadi::SX angle = if_else(trace_greater_than_ub, 0, cond_theta);

  // Check for if theta is larger than pi
  ::casadi::SX theta_larger_than_pi = angle >= M_PI - 1e-2;

  // Create relaxation TODO: This numerical precision should be an option or
  // extracted from somewhere
  ::casadi::SX cond_prec = angle > 1e-8;
  ::casadi::SX cphi = -(trace - 1) / 2;
  ::casadi::SX beta = angle * angle / (1 + cphi);
  Eigen::Vector<::casadi::SX, 3> tmp((R.diagonal().array() + cphi) * beta);
  ::casadi::SX t = if_else(cond_prec, angle / sin(angle), 1.0) / 2.0;

  // Create switching conditions for relaxation
  ::casadi::SX cond_R1 = if_else(R(2, 1) > R(1, 2), 1, -1),
               cond_R2 = if_else(R(0, 2) > R(2, 0), 1, -1),
               cond_R3 = if_else(R(1, 0) > R(0, 1), 1, -1);

  ::casadi::SX cond_tmp1 = if_else(tmp[0] > 0, sqrt(tmp[0]), 0),
               cond_tmp2 = if_else(tmp[1] > 0, sqrt(tmp[1]), 0),
               cond_tmp3 = if_else(tmp[2] > 0, sqrt(tmp[2]), 0);

  res[0] = if_else(theta_larger_than_pi, cond_R1 * cond_tmp1,
                   t * (R(2, 1) - R(1, 2)));
  res[1] = if_else(theta_larger_than_pi, cond_R2 * cond_tmp2,
                   t * (R(0, 2) - R(2, 0)));
  res[2] = if_else(theta_larger_than_pi, cond_R3 * cond_tmp3,
                   t * (R(1, 0) - R(0, 1)));

  // Copy angle to theta for return
  theta = angle;

  return res;
}

template <>
Eigen::Matrix<::casadi::SX, 3, -1> log3(
    const Eigen::Matrix<::casadi::SX, 3, 3> &R) {
  ::casadi::SX theta;
  return log3(R, theta);
}

template <>
void Jlog3(::casadi::SX &theta, const Eigen::Matrix<::casadi::SX, 3, 1> &log,
           Eigen::Matrix<::casadi::SX, 3, 3> &J) {
  ::casadi::SX alpha, diag;

  ::casadi::SX cond_prec = theta < 1e-8;
  ::casadi::SX st = sin(theta), ct = cos(theta);
  ::casadi::SX st_1mct = st / (1.0 - ct);

  alpha = if_else(cond_prec, 1.0 / 12.0 + theta * theta / 720.0,
                  1.0 / (theta * theta) - (st_1mct) / (2 * theta));

  diag = if_else(cond_prec, 0.5 * (2.0 - theta * theta / 6.0),
                 0.5 * theta * st_1mct);

  // Create result
  J.noalias() = alpha * log * log.transpose();
  J.diagonal().array() += diag;
  J += 0.5 * pinocchio::skew(log);
}

template <>
Eigen::Matrix<::casadi::SX, 6, -1> log6(
    const Eigen::Matrix<::casadi::SX, 3, 3> &R,
    const Eigen::Matrix<::casadi::SX, 3, 1> &p) {
  // Result
  Eigen::Vector<::casadi::SX, 6> res;

  // Get rotational component on Lie algebra
  ::casadi::SX theta;
  Eigen::Matrix<::casadi::SX, 3, 1> w = log3(R, theta);

  ::casadi::SX theta2 = theta * theta;

  ::casadi::SX alpha, beta;

  // Create relaxation TODO: This numerical precision should be an option or
  // extracted from somewhere
  ::casadi::SX cond_prec = theta < 1e-8;

  ::casadi::SX st = sin(theta), ct = cos(theta);

  alpha = if_else(cond_prec, 1 - theta2 / 12 - theta2 * theta2 / 720,
                  theta * st / (2 * (1 - ct)));

  beta = if_else(cond_prec, 1.0 / 12.0 + theta2 / 720,
                 1.0 / theta2 - st / (2 * theta * (1 - ct)));

  res.topRows(3).noalias() =
      alpha * p - 0.5 * w.cross(p) + (beta * w.dot(p)) * w;
  res.bottomRows(3) = w;

  return res;
}

template <>
void Jlog6(const Eigen::Matrix<::casadi::SX, 3, 3> &R,
           const Eigen::Matrix<::casadi::SX, 3, 1> &p,
           Eigen::Matrix<::casadi::SX, 6, 6> &J) {
  // Compute theta
  ::casadi::SX theta;
  Eigen::Vector<::casadi::SX, 3> w = log3(R, theta);

  // Get blocks of the jacobian
  Eigen::Matrix3<::casadi::SX> TL, TR, BL, BR;

  Jlog3(theta, w, TL);
  BR = TL;

  ::casadi::SX theta2 = theta * theta;
  ::casadi::SX beta, beta_dot_over_theta;

  ::casadi::SX cond_prec = theta < 1e-8;
  ::casadi::SX tinv = 1.0 / theta;
  ::casadi::SX tinv2 = tinv * tinv;
  ::casadi::SX st = sin(theta), ct = cos(theta);
  ::casadi::SX inv_2_2ct = 1.0 / (2 * (1.0 - ct));

  ::casadi::SX st_1mct = st / (1.0 - ct);

  beta = if_else(cond_prec, 1.0 / 12.0 + theta2 / 720.0,
                 tinv2 - st * tinv * inv_2_2ct);

  beta_dot_over_theta =
      if_else(cond_prec, 1.0 / 360.0,
              -2.0 * tinv2 * tinv2 + (1.0 + st * tinv) * tinv2 * inv_2_2ct);

  ::casadi::SX wTp = w.dot(p);
  Eigen::Vector3<::casadi::SX> v3_tmp(
      (beta_dot_over_theta * wTp) * w -
      (theta2 * beta_dot_over_theta + 2.0 * beta) * p);

  BL.noalias() = v3_tmp * w.transpose();
  BL.noalias() += beta * w * p.transpose();
  BL.diagonal().array() += wTp * beta;
  BL += pinocchio::skew(0.5 * p);

  TR.noalias() = BL * TL;
  BL.setZero();

  J << TL, TR, BL, BR;
}

}  // namespace math
}  // namespace damotion
