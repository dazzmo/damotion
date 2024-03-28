#include "planning/path_parameterisation/ua1.h"

namespace damotion {
namespace planning {
namespace path_parameterisation {

UA1ProfileGenerator::Status UA1ProfileGenerator::ComputeProfile(
    trajectory::Trajectory<double>& path, double sv0) {
    // Path rate squared ṡ²
    double sv2 = pow(sv0, 2);
    // Derivative of ṡ² with respect to s
    double dsv2 = 0;

    // Underactuated components of the dynamic coefficient vectors a, b, c
    double au = 0.0, bu = 0.0, cu = 0.0;
    double au_prev = 0.0;

    // Determine integration step size
    double ds = 1.0 / n_steps_;
    // User-defined cutoff point
    double s_cutoff = ds * cutoff_threshold_;

    // Flag to indicate if the profile exceeds the user-defined cutoff threshold
    profile_passed_cutoff_threshold_ = false;

    // Counter to increase profile
    int cnt = 0;
    bool first_iteration = true;
    n_steps_feasible_ = 0;

    for (double s = 0.0; s <= 1.0; s += ds) {
        // Evaluate the path at this point
        w_P_ = path.eval(s);
        w_dP_ = path.derivative(s, 1);
        w_ddP_ = path.derivative(s, 2);

        // Compute the path parameterised dynamic coefficient vectors a, b and c
        f_abc_.setInput({0, 1, 2}, {w_P_, w_dP_, w_ddP_});
        f_abc_.call();
        Eigen::Vector3d abc;
        au = f_abc_.getOutput(0)(0);
        bu = f_abc_.getOutput(0)(1);
        cu = f_abc_.getOutput(0)(2);

        /* Estimate the derivative of ṡ² at s */
        // If starting point
        if (first_iteration && abs(au) < 1e-1) {
            if (bu * cu > 0) {
                return Status::kNonTraversable;
            }
            // Estimate the derivative of ṡ² at this point
            dsv2 = 1.0;
            sv2 = -cu / bu;

        } else if (abs(au) <= 1e-1 && first_iteration == false) {
            // Check traversability given ṡ = √(− bᵤ / cᵤ) when aᵤ ≈ 0
            if (bu * cu > 0) {
                return Status::kNonTraversable;
            }
            if (cnt >= 2) {
                // Perform finite differencing around the point
                dsv2 = (pow(sv_[cnt - 1], 2) - pow(sv_[cnt - 2], 2)) / ds;
            } else {
                // Arbitrarily choose starting acceleration at beginning point
                dsv2 = 1.0;
            }

            sv2 = -cu / bu;
        } else {
            // Handle derivative normally
            dsv2 = -2.0 * (bu * sv2 + cu) / au;
        }

        // Assess the kinodynamic parameters
        w_qvel_ = sqrt(sv2) * w_dP_;
        for (int i = 0; i < n_; ++i) {
            if (w_qvel_[i] > qvel_max_[i] || w_qvel_[i] < -qvel_max_[i]) {
                return Status::kVelocityBoundExceeded;
            }
        }

        // Inverse dynamics torque
        w_qacc_ = 0.5 * dsv2 * w_dP_ + sv2 * w_ddP_;

        // Call inverse dynamics
        f_inv_.setInput({0, 1, 2}, {w_P_, w_qvel_, w_qacc_});
        f_inv_.call();
        w_tau_ = f_inv_.getOutput(0);

        for (int i = 0; i < n_; ++i) {
            if (i == underactuated_idx_) continue;
            if (w_tau_[i] > tau_max_[i] || w_tau_[i] < tau_min_[i]) {
                return Status::kActuatorBoundExceeded;
            }
        }

        // Add to sv trajectory
        sv_[cnt] = sqrt(sv2);

        // Integrate the profile
        // TODO: What other schemes can be used?
        sv2 += ds * dsv2;

        if (sv2 < 0.0) {
            return Status::kNonTraversable;
        }

        if (s >= s_cutoff) {
            profile_passed_cutoff_threshold_ = true;
        }

        au_prev = au;
        // Set number of feasible steps performed
        n_steps_feasible_ = cnt;
        cnt++;

        // Indicate first iteration is over
        first_iteration = false;
    }

    return Status::kSuccess;
}

// casadi::Function createZeroDynamicsCoefficients(
//     utils::casadi::PinocchioModelWrapper& wrapper, int unactuated_idx) {
//     typedef casadi::SX Scalar;
//     int nq = wrapper.model().nq;

//     // Path position, gradient and curvature
//     Scalar pos = Scalar::sym("pos", nq), grd = Scalar::sym("grd", nq),
//            crv = Scalar::sym("crv", nq);

//     // Create zero vector
//     Scalar vz = Scalar::zeros(nq);
//     Scalar a(nq), b(nq), c(nq);
//     // Set q = P, q̇ = 0, q̈ = 0 to get c = G
//     c = wrapper.rnea()(std::vector<Scalar>({pos, vz, vz}))[0];
//     // Set q = P, q̇ = 0, q̈ = P' to get a = (M(q) P' + G) - G
//     a = wrapper.rnea()(std::vector<Scalar>({pos, vz, grd}))[0] - c;
//     // Set q = P, q̇ = P', q̈ = P'' to get b = (M(q) P'' + C(q, q̇) P' + G) - G
//     b = wrapper.rnea()(std::vector<Scalar>({pos, grd, crv}))[0] - c;

//     /* Extract the underactuated components */
//     Scalar abc(3);
//     abc(0) = a(unactuated_idx);
//     abc(1) = b(unactuated_idx);
//     abc(2) = c(unactuated_idx);

//     // Create function
//     return casadi::Function(wrapper.model().name + "_abc", {pos, grd, crv},
//                             {abc}, {"pos", "grd", "crv"}, {"abc"});
// };

}  // namespace path_parameterisation
}  // namespace planning
}  // namespace damotion