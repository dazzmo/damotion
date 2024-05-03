#ifndef MODEL_FRAME_H
#define MODEL_FRAME_H

#include "damotion/utils/eigen_wrapper.h"

namespace damotion {
namespace model {

namespace symbolic {

class TargetFrame {
 public:
  friend class model::TargetFrame;

  TargetFrame() { x_.resize(3); }
  ~TargetFrame() = default;

  /**
   * @brief Position of the frame in the given reference frame
   *
   * @return const ::casadi::SX&
   */
  const casadi::SX &pos() { return x_[0]; }
  /**
   * @brief Velocity of the frame in the given reference frame
   *
   * @return const ::casadi::SX&
   */
  const casadi::SX &vel() { return x_[1]; }
  /**
   * @brief Acceleration of the frame in the given reference frame
   *
   * @return const ::casadi::SX&
   */
  const casadi::SX &acc() { return x_[2]; }

  void UpdateState(const ::casadi::SX &qpos, const ::casadi::SX &qvel,
                   const ::casadi::SX &qacc) {
    x_ = f_(::casadi::SXVector({qpos, qvel, qacc}));
  }

 protected:
  void SetFunction(const ::casadi::Function &f) { f_ = f; }

 private:
  // Current frame state for the symbolic function
  ::casadi::SXVector x_;

  // Function to compute the state of the frame
  ::casadi::Function f_;
};

}  // namespace symbolic

/**
 * @brief Target frame of interest on the model, such as an end-effector
 * position or reference frame such as the centre of mass.
 *
 */
class TargetFrame {
 public:
  TargetFrame() = default;
  ~TargetFrame() = default;

  /**
   * @brief Construct a new Target Frame object from its symbolic equivalent
   *
   * @param sym
   */
  TargetFrame(const symbolic::TargetFrame &sym) {
    // Wrap function
    SetFunction(sym.f_);
  }

  /**
   * @brief Position of the frame in the given reference frame
   *
   * @return const Eigen::Ref<const Eigen::VectorXd>&
   */
  const Eigen::Ref<const Eigen::VectorXd> &pos() {
    return f_wrapper_.getOutput(0);
  }
  /**
   * @brief Velocity of the frame in the given reference frame
   *
   * @return const Eigen::Ref<const Eigen::VectorXd>&
   */
  const Eigen::Ref<const Eigen::VectorXd> &vel() {
    return f_wrapper_.getOutput(1);
  }
  /**
   * @brief Acceleration of the frame in the given reference frame
   *
   * @return const Eigen::Ref<const Eigen::VectorXd>&
   */
  const Eigen::Ref<const Eigen::VectorXd> &acc() {
    return f_wrapper_.getOutput(2);
  }

  void UpdateState(const Eigen::VectorXd &qpos, const Eigen::VectorXd &qvel,
                   const Eigen::VectorXd &qacc) {
    f_wrapper_.call({qpos, qvel, qacc});
  }

 protected:
  void SetFunction(const ::casadi::Function &f) { f_wrapper_ = f; }

 private:
  // Wrapper for the function of the symbolic function
  utils::casadi::FunctionWrapper<Eigen::VectorXd> f_wrapper_;
};

}  // namespace model
}  // namespace damotion

#endif /* MODEL_FRAME_H */
