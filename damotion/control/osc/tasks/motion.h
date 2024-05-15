#ifndef TASKS_MOTION_H
#define TASKS_MOTION_H

#include "damotion/control/osc/tasks/task.h"

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

  using SharedPtr = std::shared_ptr<MotionTask>;

  MotionTask(const std::string &name, const int &xdim, const int &vdim)
      : Task(name, xdim, vdim) {}

  /**
   * @brief Compute the desired tracking task acceleration as a PD error
   * metric on the task position and velocity errors
   *
   * @return Eigen::VectorXd
   */
  virtual void ComputeMotionError() = 0;

 private:
};

class PositionTask : public MotionTask {
 public:
  PositionTask() = default;
  ~PositionTask() = default;

  PositionTask(const std::string &name, const casadi::SX &xpos,
               const casadi::SX &xvel, const casadi::SX &xacc)
      : MotionTask(name, 3, 3) {
    assert(xpos.size1() == 3 && xvel.size1() == 3 && xacc.size1() == 3 &&
           "Symbolic vectors are incorrect size for PositionTask!");
  }

  struct Reference {
    Eigen::Vector3d x;
    Eigen::Vector3d v;
  };

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

  /**
   * @brief Construct a new Orientation Task object
   *
   * @param xpos Quaternion representing the orientation of the task frame
   * @param xvel
   * @param xacc
   */
  OrientationTask(const std::string &name, const casadi::SX &xpos,
                  const casadi::SX &xvel, const casadi::SX &xacc)
      : MotionTask(name, 4, 3) {
    assert(xpos.size1() == 4 && xvel.size1() == 3 && xacc.size1() == 3 &&
           "Symbolic vectors are incorrect size for OrientationTask!");
  }

  struct Reference {
    Eigen::Quaterniond q;
    Eigen::Vector3d w;
  };

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

  /**
   * @brief Construct a new PoseTask object
   *
   * @param xpos Vector rerpresenting the state (position and orientation) of
   * the task frame (position vector and quaternion representation)
   * @param xvel
   * @param xacc
   */
  PoseTask(const std::string &name, const casadi::SX &xpos,
           const casadi::SX &xvel, const casadi::SX &xacc)
      : MotionTask(name, 7, 6) {
    assert(xpos.size1() == 7 && xvel.size1() == 6 && xacc.size1() == 6 &&
           "Symbolic vectors are incorrect size for PoseTask!");
  }

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
