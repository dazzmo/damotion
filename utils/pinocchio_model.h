#ifndef UTILS_PINOCCHIO_MODEL_H
#define UTILS_PINOCCHIO_MODEL_H

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/autodiff/casadi/utils/static-if.hpp>
#include <pinocchio/autodiff/casadi/math/quaternion.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "utils/eigen_wrapper.h"

namespace casadi_utils {

class PinocchioModelWrapper {
   public:
    // Define casadi autodiff type
    using AD = casadi::SXElem;

    PinocchioModelWrapper() = default;
    PinocchioModelWrapper(pinocchio::Model &model) { *this = model; }

    PinocchioModelWrapper &operator=(pinocchio::Model model);

    /**
     * @brief Returns a casadi::Function that computes the forward-dynamics of
     * the system through the Articulated Body Algorithm (ABA)
     *
     * @return casadi::Function
     */
    casadi::Function aba();

    /**
     * @brief Returns a casadi::Function that computes the inverse-dynamics of
     * the system through the Recursive Newton-Euler Algorithm (ABA)
     *
     * @return casadi::Function
     */
    casadi::Function rnea();

    /**
     * @brief End-effector data for a given point on the system, includes data
     * such as its Jacobian, constraint-subspace and acceleration
     *
     */
    struct EndEffector {
        /**
         * @brief Pose of the end-effector in SE3 space, useful for converting
         * to se3 for error computation
         *
         */
        pinocchio::SE3Tpl<casadi::Matrix<AD>> pose;

        /**
         * @brief Function that computes the end-effector position/orientation
         * (x), velocity and acceleration. A function with inputs (q, v, a) and
         * output (x, dx, ddx)
         *
         */
        casadi::Function x;

        /**
         * @brief End-effector Jacobian \f$ J(q) \f$
         *
         */
        casadi::Function J;

        /**
         * @brief Constraint motion subspace (i.e. basis for the constraint
         * forces that act on the end-effector in the operational frame)
         *
         */
        Eigen::Matrix<double, 6, -1> S;
    };

    /**
     * @brief End-effector at index i in the end-effector vector
     *
     * @param i
     * @return EndEffector&
     */
    EndEffector &end_effector(int i) { return ee_[i]; }

    /**
     * @brief The index of the end-effector given by name in the end-effector
     * vector
     *
     * @param name Name of the end effector
     * @return const int&
     */
    const int &end_effector_idx(const std::string &name) {
        return ee_idx_[name];
    }

    void addEndEffector(const std::string &frame_name);

    pinocchio::ModelTpl<casadi::Matrix<AD>> &model() { return model_; }
    pinocchio::DataTpl<casadi::Matrix<AD>> &data() { return data_; }

   private:
    pinocchio::ModelTpl<casadi::Matrix<AD>> model_;
    pinocchio::DataTpl<casadi::Matrix<AD>> data_;

    std::vector<EndEffector> ee_;
    std::unordered_map<std::string, int> ee_idx_;
};

}  // namespace casadi_utils

#endif /* UTILS_PINOCCHIO_MODEL_H */
