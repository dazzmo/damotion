#ifndef PATH_PARAMETERISATION_UA1_H
#define PATH_PARAMETERISATION_UA1_H

#include "common/trajectory/trajectory.h"
#include "utils/eigen_wrapper.h"
#include "utils/pinocchio_model.h"

namespace damotion {
namespace planning {
namespace path_parameterisation {

class UA1ProfileGenerator {
   public:
    UA1ProfileGenerator(int n, const int& underactuated_idx, const int n_steps)
        : n_(n), underactuated_idx_(underactuated_idx), n_steps_(n_steps) {
        cutoff_threshold_ = 0;

        sv_.resize(n_steps_ + 1);
        s_.resize(n_steps_ + 1);

        double s = 0.0;
        double ds = 1.0 / (n_steps);
        for (int i = 0; i <= n_steps; ++i) {
            s_[i] = s;
            s += ds;
        }

        w_P_.resize(n);
        w_dP_.resize(n);
        w_ddP_.resize(n);

        w_qvel_.resize(n);
        w_qacc_.resize(n);
        w_tau_.resize(n);

        qvel_max_.resize(n);
        tau_min_.resize(n);
        tau_max_.resize(n);

        qvel_max_.setConstant(std::numeric_limits<double>::infinity());
        tau_min_.setConstant(-std::numeric_limits<double>::infinity());
        tau_max_.setConstant(std::numeric_limits<double>::infinity());

        // Kinodynamic parameters
        Eigen::VectorXd qvel(n);
        Eigen::VectorXd tau(n);

    }

    enum class Status {
        kSuccess = 0,
        kNonTraversable,
        kActuatorBoundExceeded,
        kVelocityBoundExceeded
    };

    /**
     * @brief Computes the path paramterisation for the given path defined over
     * the domain \f$ s \in [0, 1] \f$ and system with underactuated index and
     * starting path velocity.
     *
     * @param path
     * @param system
     * @return Status
     */
    Status ComputeProfile(trajectory::Trajectory<double>& path, double sv0);

    /**
     * @brief Set the number of integration steps the parameterisation needs to
     * perform to be considered a traversable path
     *
     * @param threshold
     */
    void SetCutoffThreshold(const int& threshold) {
        cutoff_threshold_ = threshold;
    }

    /**
     * @brief Flag to indicate whether the current profile computed by
     * ComputeProfile() exceeds the user-defined threshold \f$ s^\star = h
     * \Delta s \f$ where \f$ h \f$ is the number of integration steps.
     *
     * @return true
     * @return false
     */
    bool ProfileExceedsCutoffPoint() {
        return profile_passed_cutoff_threshold_;
    }

    /**
     * @brief Starting iterator to the feasible section of the s vector
     *
     * @return std::vector<double>::iterator
     */
    std::vector<double>::iterator s_begin() { return s_.begin(); }

    /**
     * @brief Ending iterator to the feasible section of the s vector
     *
     * @return std::vector<double>::iterator
     */
    std::vector<double>::iterator s_end() {
        return s_.begin() + n_steps_feasible_;
    }

    /**
     * @brief Starting iterator to the feasible section of the sv vector
     *
     * @return std::vector<double>::iterator
     */
    std::vector<double>::iterator sv_begin() { return sv_.begin(); }

    /**
     * @brief Ending iterator to the feasible section of the sv vector
     *
     * @return std::vector<double>::iterator
     */
    std::vector<double>::iterator sv_end() {
        return sv_.begin() + n_steps_feasible_;
    }

    /**
     * @brief Vector of the feasible trajectory for \f$ \dot{s}(s) \f$
     *
     * @return std::vector<double>
     */
    std::vector<double> sv() {
        // Depending on how many steps were feasible
        std::vector<double> sv = std::vector<double>(sv_begin(), sv_end());
        return sv;
    }

    /**
     * @brief Vector of the feasible duration for the trajectory for
     * \f$ \dot{s}(s) \f$
     *
     * @return std::vector<double>
     */
    std::vector<double> s() {
        // Depending on how many steps were feasible
        std::vector<double> s = std::vector<double>(s_begin(), s_end());

        return s;
    }

    void setMaximumVelocity(const Eigen::VectorXd& qvel) { qvel_max_ = qvel; }
    Eigen::VectorXd& getMaximumVelocity() { return qvel_max_; }

    void setMaximumActuation(const Eigen::VectorXd& tau) { tau_max_ = tau; }
    Eigen::VectorXd& getMaximumActuation() { return tau_max_; }

    void setMinimumActuation(const Eigen::VectorXd& tau) { tau_min_ = tau; }
    Eigen::VectorXd& getMinimumActuation() { return tau_min_; }

    void setZeroDynamicsCoefficientFunction(
        utils::casadi::FunctionWrapper& fun) {
        f_abc_ = fun;
    }
    utils::casadi::FunctionWrapper& getZeroDynamicsCoefficientFunction() {
        return f_abc_;
    }

    void setInverseDynamicsFunction(utils::casadi::FunctionWrapper& fun) {
        f_inv_ = fun;
    }
    utils::casadi::FunctionWrapper& getInverseDynamicsFunction() {
        return f_inv_;
    }

   private:
    int n_;
    int n_steps_;
    int cutoff_threshold_;
    int underactuated_idx_;
    bool profile_passed_cutoff_threshold_ = false;
    int n_steps_feasible_;

    // Trajectory for the path-rate profile
    std::vector<double> sv_;
    // Discretisation over s
    std::vector<double> s_;

    /* Work vectors */
    Eigen::VectorXd w_P_;    // Path
    Eigen::VectorXd w_dP_;   // Path gradient
    Eigen::VectorXd w_ddP_;  // Path curvature

    Eigen::VectorXd w_qvel_;  // Generalised velocity
    Eigen::VectorXd w_qacc_;  // Generalised acceleration
    Eigen::VectorXd w_tau_;   // Generalised actuation

    Eigen::VectorXd qvel_max_;  // Maximum velocity

    Eigen::VectorXd tau_max_;  // Maximum actuation
    Eigen::VectorXd tau_min_;  // Minimum actuation

    /* Function wrappers to compute kinodynamic quantities */
    utils::casadi::FunctionWrapper f_abc_;
    utils::casadi::FunctionWrapper f_inv_;

};

casadi::Function createZeroDynamicsCoefficients(
    utils::casadi::PinocchioModelWrapper& wrapper, int unactuated_idx);

}  // namespace path_parameterisation
}  // namespace planning
}  // namespace damotion

#endif /* PATH_PARAMETERISATION_UA1_H */
