#ifndef UTILS_PINOCCHIO_MODEL_H
#define UTILS_PINOCCHIO_MODEL_H

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/autodiff/casadi.hpp>
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
     * @brief Sets the constraint subspace for the end-effector such that
     * constraint forces will be applied in the dimensions indicated by S
     *
     * @param i
     * @param S
     */
    void setEndEffectorConstraintSubspace(int i,
                                          const Eigen::Vector<double, 6> &S);
                                          
    void addEndEffector(const std::string &frame_name);

    casadi::Function &end_effector(int i) { return ee_[i]; }
    casadi::Function &end_effector_jac(int i) { return ee_jac_[i]; }

    // Add end-effector

    pinocchio::ModelTpl<casadi::Matrix<AD>> &model() { return model_; }
    pinocchio::DataTpl<casadi::Matrix<AD>> &data() { return data_; }

   private:
    pinocchio::ModelTpl<casadi::Matrix<AD>> model_;
    pinocchio::DataTpl<casadi::Matrix<AD>> data_;

    // Vector of end effectors to consider for force distibution in rigid-body
    // algorithms
    std::vector<casadi::Function> ee_;
    std::vector<casadi::Function> ee_jac_;
    std::vector<Eigen::Vector<double, 6>> ee_constraint_subspace_;
};

}  // namespace casadi_utils

#endif /* UTILS_PINOCCHIO_MODEL_H */
