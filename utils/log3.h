#ifndef UTILS_POSE_H
#define UTILS_POSE_H

#include <casadi/casadi.hpp>
#include <pinocchio/autodiff/casadi.hpp>

namespace damotion {

/**
 * @brief Symbolic function to compute the logarithm map for a matrix R in SO3.
 * Based on pinocchio's implementation of the function
 * (pinocchio/spatial/log.hxx) which includes limits for numerical stability.
 *
 * @param R A matrix in SO3 to be mapped to the Lie Algebra of so3
 * @return Eigen::Matrix<casadi::SX, 3, -1>
 */
Eigen::Matrix<casadi::SX, 3, -1> log3(
    const Eigen::Matrix<casadi::SX, 3, 3> &R) {
    Eigen::Vector<casadi::SX, 3> res;

    // Upper and lower bounds for the trace of R
    casadi::SX ub = casadi::SX(3);
    casadi::SX lb = casadi::SX(-1);

    // Compute trace of matrix
    casadi::SX tr = R.trace();

    // Create conditions for checking bounds
    casadi::SX trace_greater_than_ub = tr >= ub;
    casadi::SX trace_less_than_lb = tr <= lb;

    // Determine trace
    casadi::SX cond_trace = if_else(trace_less_than_lb, lb, R.trace());
    casadi::SX trace = if_else(trace_greater_than_ub, ub, cond_trace);

    // Determine theta based on given conditions
    casadi::SX cond_theta =
        if_else(trace_less_than_lb, M_PI, acos((tr - 1) / 2));
    casadi::SX theta = if_else(trace_greater_than_ub, 0, cond_theta);

    // Check for if theta is larger than pi
    casadi::SX theta_larger_than_pi = theta >= M_PI - 1e-2;

    // Create relaxation TODO: This numerical precision should be an option or
    // extracted from somewhere
    casadi::SX cond_prec = theta > 1e-8;
    casadi::SX cphi = -(trace - 1) / 2;
    casadi::SX beta = theta * theta / (1 + cphi);
    Eigen::Vector<casadi::SX, 3> tmp((R.diagonal().array() + cphi) * beta);
    casadi::SX t = if_else(cond_prec, theta / sin(theta), 1.0) / 2.0;

    // Create switching conditions for relaxation
    casadi::SX cond_R1 = if_else(R(2, 1) > R(1, 2), 1, -1),
               cond_R2 = if_else(R(0, 2) > R(2, 0), 1, -1),
               cond_R3 = if_else(R(1, 0) > R(0, 1), 1, -1);

    casadi::SX cond_tmp1 = if_else(tmp[0] > 0, sqrt(tmp[0]), 0),
               cond_tmp2 = if_else(tmp[1] > 0, sqrt(tmp[1]), 0),
               cond_tmp3 = if_else(tmp[2] > 0, sqrt(tmp[2]), 0);

    res[0] = if_else(theta_larger_than_pi, cond_R1 * cond_tmp1,
                     t * (R(2, 1) - R(1, 2)));
    res[1] = if_else(theta_larger_than_pi, cond_R2 * cond_tmp2,
                     t * (R(0, 2) - R(2, 0)));
    res[2] = if_else(theta_larger_than_pi, cond_R3 * cond_tmp3,
                     t * (R(1, 0) - R(0, 1)));

    return res;
}

}  // namespace damotion

#endif /* UTILS_POSE_H */
