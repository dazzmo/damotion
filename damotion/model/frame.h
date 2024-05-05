#ifndef MODEL_FRAME_H
#define MODEL_FRAME_H

#include "damotion/utils/eigen_wrapper.h"

namespace damotion {
namespace model {

/**
 * @brief Target frame of interest of a kinematic model, such as an end-effector
 * position or reference frame such as the centre of mass.
 *
 */
class TargetFrame {
 public:
  TargetFrame() = default;
  ~TargetFrame() = default;

  using SharedPtr = std::shared_ptr<TargetFrame>;
  using UniquePtr = std::unique_ptr<TargetFrame>;

  /**
   * @brief Position of the frame in the given reference frame
   *
   * @return const Eigen::Ref<const Eigen::VectorXd>&
   */
  Eigen::Ref<const Eigen::VectorXd> pos() { return f_->getOutput(0); }
  /**
   * @brief Velocity of the frame in the given reference frame
   *
   * @return const Eigen::Ref<const Eigen::VectorXd>&
   */
  Eigen::Ref<const Eigen::VectorXd> vel() { return f_->getOutput(1); }
  /**
   * @brief Acceleration of the frame in the given reference frame
   *
   * @return const Eigen::Ref<const Eigen::VectorXd>&
   */
  Eigen::Ref<const Eigen::VectorXd> acc() { return f_->getOutput(2); }

  void UpdateState(const Eigen::VectorXd &qpos, const Eigen::VectorXd &qvel,
                   const Eigen::VectorXd &qacc) {
    f_->call({qpos, qvel, qacc});
  }

  /**
   * @brief Set function to compute the frame position, velocity and
   * acceleration.
   *
   * @param f
   */
  void SetFunction(const common::Function<Eigen::VectorXd>::SharedPtr &f) {
    f_ = f;
  }

 private:
  // Wrapper for the function of the symbolic function
  common::Function<Eigen::VectorXd>::SharedPtr f_;
};

namespace symbolic {

class TargetFrame {
 public:
  friend class model::TargetFrame;

  /**
   * @brief Construct a new Target Frame object based on the function f.
   *
   * @param f
   */
  TargetFrame(const ::casadi::Function &f) {
    f_ = f;
    x_.resize(3);
  }
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

  /**
   * @brief Create a Frame object from the symbolic representation
   *
   * @return model::TargetFrame::SharedPtr
   */
  model::TargetFrame::SharedPtr CreateFrame() {
    if (target_frame_made_) {
      target_frame_ = std::make_shared<model::TargetFrame>();
      // Create function wrapper
      utils::casadi::FunctionWrapper<Eigen::VectorXd>::SharedPtr f =
          std::make_shared<utils::casadi::FunctionWrapper<Eigen::VectorXd>>(
              this->f_);
      target_frame_->SetFunction(f);
    }
    return target_frame_;
  }

 protected:
  void SetFunction(const ::casadi::Function &f) { f_ = f; }

 private:
  bool target_frame_made_ = false;

  // Current frame state for the symbolic function
  ::casadi::SXVector x_;
  // Function to compute the state of the frame
  ::casadi::Function f_;

  // Shared pointer to resulting target frame created by the symbolic
  // representation
  model::TargetFrame::SharedPtr target_frame_ = nullptr;
};

}  // namespace symbolic

}  // namespace model
}  // namespace damotion

#endif /* MODEL_FRAME_H */
