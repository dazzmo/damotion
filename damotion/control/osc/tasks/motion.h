#ifndef TASKS_MOTION_H
#define TASKS_MOTION_H

#include "damotion/control/osc/tasks/task.h"

namespace opt = damotion::optimisation;
namespace sym = damotion::symbolic;
namespace utils = damotion::utils;

namespace damotion {
namespace control {
namespace osc {

/**
 * @brief Motion task for operational space control, typically used for tracking
 * of a frame
 *
 */
class MotionTask : public Task {
 public:
  MotionTask() = default;
  ~MotionTask() = default;

  MotionTask(const std::string &name) : Task(name) {}

  /**
   * @brief Position of the frame the motion task is based on
   *
   * @return Eigen::VectorXd
   */
  virtual Eigen::VectorXd pos() = 0;

  /**
   * @brief Velocity of the frame the motion task is based on
   *
   * @return Eigen::VectorXd
   */
  virtual Eigen::VectorXd vel() = 0;

  /**
   * @brief Acceleration of the frame the motion task is based on
   *
   * @return Eigen::VectorXd
   */
  virtual Eigen::VectorXd acc() = 0;

  /**
   * @brief Set the frame to base the motion task on
   *
   * @param frame
   */
  void SetFrame(const std::shared_ptr<model::TargetFrame> &frame) {
    frame_ = frame;
  }

  /**
   * @brief The TargetFrame the motion task is designed for
   *
   * @return const TargetFrame&
   */
  model::TargetFrame &Frame() { return *frame_; }

  /**
   * @brief Compute the desired tracking task acceleration as a PD error
   * metric on the task position and velocity errors
   *
   * @return Eigen::VectorXd
   */
  virtual void ComputeMotionError() = 0;

 private:
  // Target frame the motion task is associated with
  std::shared_ptr<model::TargetFrame> frame_;
};

class PositionTask : public MotionTask {
 public:
  PositionTask() = default;
  ~PositionTask() = default;

  struct Reference {
    Eigen::Vector3d x;
    Eigen::Vector3d v;
  };

  PositionTask(const std::string &name) : MotionTask(name) { ResizeTask(3); }

  Eigen::VectorXd pos() override { return Frame().pos().topRows(3); }
  Eigen::VectorXd vel() override { return Frame().vel().topRows(3); }
  Eigen::VectorXd acc() override { return Frame().acc().topRows(3); }

  void ComputeMotionError();

  const Reference &GetReference() { return ref_; }
  void SetReference(const Eigen::Vector3d &x, const Eigen::Vector3d &v) {
    ref_.x = x;
    ref_.v = v;
  }

 private:
  Reference ref_;
};

class OrientationTask : public MotionTask {
 public:
  OrientationTask() = default;
  ~OrientationTask() = default;

  struct Reference {
    Eigen::Quaterniond q;
    Eigen::Vector3d w;
  };

  OrientationTask(const std::string &name) : MotionTask(name) { ResizeTask(3); }

  Eigen::VectorXd pos() override { return Frame().pos().bottomRows(4); }
  Eigen::VectorXd vel() override { return Frame().vel().bottomRows(3); }
  Eigen::VectorXd acc() override { return Frame().acc().bottomRows(3); }

  /**
   * @brief Compute the desired tracking task acceleration as a PD error
   * metric on the task position and velocity errors
   *
   * @return Eigen::VectorXd
   */
  void ComputeMotionError();

  const Reference &GetReference() { return ref_; }
  void SetReference(const Eigen::Quaterniond &q, const Eigen::Vector3d &w) {
    ref_.q = q;
    ref_.w = w;
  }

 private:
  Reference ref_;
};

class PoseTask : public MotionTask {
 public:
  PoseTask() = default;
  ~PoseTask() = default;

  struct Reference {
    Eigen::Vector3d x;
    Eigen::Quaterniond q;
    Eigen::Vector3d v;
    Eigen::Vector3d w;
  };

  PoseTask(const std::string &name) : MotionTask(name) { ResizeTask(6); }

  Eigen::VectorXd pos() override { return Frame().pos(); }
  Eigen::VectorXd vel() override { return Frame().vel(); }
  Eigen::VectorXd acc() override { return Frame().acc(); }

  /**
   * @brief Compute the desired tracking task acceleration as a PD error
   * metric on the task position and velocity errors
   *
   * @return Eigen::VectorXd
   */
  void ComputeMotionError();

  const Reference &GetReference() { return ref_; }
  void SetReference(const Eigen::Vector3d &x, const Eigen::Vector3d &v,
                    const Eigen::Quaterniond &q, const Eigen::Vector3d &w) {
    ref_.x = x;
    ref_.q = q;
    ref_.v = v;
    ref_.w = w;
  }

 private:
  Reference ref_;
};

}  // namespace osc
}  // namespace control
}  // namespace damotion

#endif /* TASKS_MOTION_H */
