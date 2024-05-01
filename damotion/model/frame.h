#ifndef MODEL_FRAME_H
#define MODEL_FRAME_H

#include "damotion/utils/eigen_wrapper.h"

namespace damotion {
namespace model {
/**
 * @brief Target frame of interest on the model, such as an end-effector
 * position or reference frame such as the centre of mass.
 *
 */
class TargetFrame {
 public:
  TargetFrame() { x_.resize(3); }
  ~TargetFrame() = default;

  /**
   * @brief Position of the frame in the given reference frame
   *
   * @return const ::casadi::SX&
   */
  const ::casadi::SX &pos_sym() { return x_[0]; }
  /**
   * @brief Velocity of the frame in the given reference frame
   *
   * @return const ::casadi::SX&
   */
  const ::casadi::SX &vel_sym() { return x_[1]; }
  /**
   * @brief Acceleration of the frame in the given reference frame
   *
   * @return const ::casadi::SX&
   */
  const ::casadi::SX &acc_sym() { return x_[2]; }

  /**
   * @brief \copydoc pos_sym()
   *
   * @return const ::casadi::SX&
   */
  const Eigen::VectorXd &pos() {
    pos_ = f_wrapper_.getOutput(0);
    return pos_;
  }
  /**
   * @brief \copydoc vel_sym()
   *
   * @return const ::casadi::SX&
   */
  const Eigen::VectorXd &vel() {
    vel_ = f_wrapper_.getOutput(1);
    return vel_;
  }
  /**
   * @brief \copydoc acc_sym()
   *
   * @return const ::casadi::SX&
   */
  const Eigen::VectorXd &acc() {
    acc_ = f_wrapper_.getOutput(2);
    return acc_;
  }

  void UpdateState(const ::casadi::SX &qpos, const ::casadi::SX &qvel,
                   const ::casadi::SX &qacc) {
    x_ = f_(::casadi::SXVector({qpos, qvel, qacc}));
  }

  void UpdateState(const Eigen::VectorXd &qpos, const Eigen::VectorXd &qvel,
                   const Eigen::VectorXd &qacc) {
    f_wrapper_.setInput(0, qpos.data());
    f_wrapper_.setInput(1, qvel.data());
    f_wrapper_.setInput(2, qacc.data());
    f_wrapper_.call();
  }

 protected:
  void SetFunction(const ::casadi::Function &f) {
    f_ = f;
    f_wrapper_ = f_;
  }

 private:
  // Current frame state for the symbolic function
  ::casadi::SXVector x_;

  Eigen::VectorXd pos_;
  Eigen::VectorXd vel_;
  Eigen::VectorXd acc_;

  // Function to compute the state of the frame
  ::casadi::Function f_;
  // Wrapper for the function of the symbolic function
  utils::casadi::FunctionWrapper f_wrapper_;
};
}  // namespace model
}  // namespace damotion

#endif /* MODEL_FRAME_H */
