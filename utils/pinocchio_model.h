#ifndef UTILS_PINOCCHIO_MODEL_H
#define UTILS_PINOCCHIO_MODEL_H

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/autodiff/casadi/math/quaternion.hpp>
#include <pinocchio/autodiff/casadi/utils/static-if.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "utils/eigen_wrapper.h"
#include "utils/log.h"

namespace damotion {
namespace utils {
namespace casadi {

class PinocchioModelWrapper {
   public:
    // Define casadi autodiff type
    using AD = ::casadi::SXElem;

    PinocchioModelWrapper() = default;
    PinocchioModelWrapper(pinocchio::Model &model) { *this = model; }

    PinocchioModelWrapper &operator=(pinocchio::Model model);

    /**
     * @brief Returns a casadi::Function that computes the forward-dynamics of
     * the system through the Articulated Body Algorithm (ABA)
     *
     * @return casadi::Function
     */
    ::casadi::Function aba();

    /**
     * @brief Returns a casadi::Function that computes the inverse-dynamics of
     * the system through the Recursive Newton-Euler Algorithm (ABA)
     *
     * @return casadi::Function
     */
    ::casadi::Function rnea();

    /**
     * @brief End-effector data for a given point on the system, includes data
     * such as its Jacobian, constraint-subspace and acceleration
     *
     */
    struct EndEffector {
        /**
         * @brief Function that computes the end-effector position/orientation
         * (x), velocity and acceleration. A function with inputs (q, v, a) and
         * output (x, dx, ddx)
         *
         */
        ::casadi::Function x;

        /**
         * @brief End-effector Jacobian \f$ J(q) \f$
         *
         */
        ::casadi::Function J;
    };

    /**
     * @brief End-effector at index i in the end-effector vector
     *
     * @param i
     * @return EndEffector&
     */
    EndEffector &end_effector(int i) { return ee_[i]; }

    /**
     * @brief End-effector with name 'name' in the end-effector vector
     *
     * @param i
     * @return EndEffector&
     */
    EndEffector &end_effector(const std::string &name) {
        return ee_[end_effector_idx(name)];
    }

    /**
     * @brief The index of the end-effector given by name in the end-effector
     * vector
     *
     * @param name Name of the end effector
     * @return const int
     */
    const int end_effector_idx(const std::string &name) {
        if (ee_idx_.find(name) == ee_idx_.end()) {
            std::cout << "End effector with name " << name
                      << " is not included in this model!\n";
            return -1;
        }
        return ee_idx_[name];
    }

    void addEndEffector(const std::string &frame_name);

    pinocchio::ModelTpl<::casadi::Matrix<AD>> &model() { return model_; }
    pinocchio::DataTpl<::casadi::Matrix<AD>> &data() { return data_; }

   private:
    pinocchio::ModelTpl<::casadi::Matrix<AD>> model_;
    pinocchio::DataTpl<::casadi::Matrix<AD>> data_;

    std::vector<EndEffector> ee_;
    std::unordered_map<std::string, int> ee_idx_;
};

/**
 * @brief Computes the error of two points in SE3 through use of the Lie algebra
 * se3 to provide a 6-dimensional vector indicative of the pose error. The
 * translational error uses the typicall Euclidean distance, whereas the
 * rotational component makes use of the log3() function to determine the
 * rotational difference.
 *
 * @param p0 The first pose
 * @param p1 The second pose
 * @return Eigen::Vector<T, 6>
 */
template <typename T>
Eigen::Vector<T, 6> poseError(const pinocchio::SE3Tpl<T> &p0,
                              const pinocchio::SE3Tpl<T> &p1) {
    // Create error vector in se3
    Eigen::Vector<T, 6> err;
    // Translational error
    err.topRows(3) = p0.translation() - p1.translation();
    // Compute difference in pose rotation
    Eigen::Matrix3<T> Rd = p0.rotation().transpose() * p1.rotation();
    // Compute difference in so3 by logarithm map
    err.bottomRows(3) = damotion::log3(Rd);

    return err;
}

}  // namespace casadi
}  // namespace utils
}  // namespace damotion

#endif /* UTILS_PINOCCHIO_MODEL_H */
