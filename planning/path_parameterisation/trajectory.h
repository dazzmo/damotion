#ifndef PATH_PARAMETERISATION_TRAJECTORY_H
#define PATH_PARAMETERISATION_TRAJECTORY_H

#include "common/log.h"
#include "common/trajectory/trajectory.h"

namespace damotion {
namespace planning {
namespace path_parameterisation {

/**
 * @brief Creates the discrete trajectory for the profile \f$ \dot{s}(s) \f$
 * over the interval given by s to some resolution ds
 *
 * @param s
 * @param sv
 * @param ds
 * @return trajectory::DiscreteTrajectory<double>
 */
trajectory::DiscreteTrajectory<double> GetPathVelocityProfile(
    std::vector<double> &s, std::vector<double> &sv, const double &ds = 0.01) {
    // Create vector for storing entries
    std::vector<VectorXd> sv_trajectory;
    std::vector<double> s_trajectory;

    int cnt = 0;
    double s0 = s[0];
    double s1 = s[1];
    // The node of sv that the current point is in front of
    double sv0 = sv[0];
    // The node of sv that the current point is behind
    double sv1 = sv[1];

    double sf = s.back();

    VectorXd x(1);

    double si = s0;
    double svi = sv0;

    while (si <= sf) {
        // Locate correct segment within the discrete profile
        while (si > s1) {
            cnt++;

            s0 = s[cnt];
            s1 = s[cnt + 1];

            sv0 = sv[cnt];
            sv1 = sv[cnt + 1];
        }

        // Determine how far into the segment the point is
        double tau = (si - s0) / (s1 - s0);
        // Linearly interpolate to get point
        svi = sqrt(pow(sv0, 2) * (1.0 - tau) + pow(sv1, 2) * tau);

        // Append next entry to trajectory
        x << svi;
        sv_trajectory.push_back(x);
        s_trajectory.push_back(si);

        // Move along s
        si += ds;
    }

    return trajectory::DiscreteTrajectory<double>(sv_trajectory, s_trajectory);
}

/**
 * @brief Generate a trajectory with resolution n for representing the
 * trajectory x(t) from P(s) and sv(s). Evaluates for the duration of the
 * provided profile.
 *
 * @param sv
 * @param path
 * @param dt Time resolution of the resulting profile
 * @return trajectory::DiscreteTrajectory<double>
 */
trajectory::DiscreteTrajectory<double> GetTrajectoryFromPathParameterisation(
    std::vector<double> &s, std::vector<double> &sv,
    trajectory::Trajectory<double> &path, const double &dt = 0.01) {
    // Create vector for storing entries
    std::vector<VectorXd> x_trajectory;
    std::vector<double> t_trajectory;

    VectorXd x(2 * path.begin().size());

    // Compute profile
    double t = 0.0;
    double dti = 0.0;

    for (int i = 0; i < s.size(); i++) {
        if (i == 0) {
            x << path.eval(s[i]), sv[i] * path.derivative(s[i], 1);

            x_trajectory.push_back(x);
            t_trajectory.push_back(t);

            // Skip values until path velocity is non-zero
            while (sv[i] < std::numeric_limits<double>::epsilon()) {
                i++;
            }

            continue;

        } else {
            double ds = s[i] - s[i - 1];
            dti = 2.0 * ds / (sv[i] + sv[i - 1]);
        }

        t += dti;

        // Add to discrete profile
        x << path.eval(s[i]), sv[i] * path.derivative(s[i], 1);

        x_trajectory.push_back(x);
        t_trajectory.push_back(t);
    }

    // Refine trajectory to resolution

    return trajectory::DiscreteTrajectory<double>(x_trajectory, t_trajectory);
}

}  // namespace path_parameterisation
}  // namespace planning
}  // namespace damotion

#endif /* PATH_PARAMETERISATION_TRAJECTORY_H */
