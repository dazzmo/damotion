#ifndef TASKS_CONTACT_H
#define TASKS_CONTACT_H

#include "damotion/control/osc/tasks/motion.h"

namespace damotion {
namespace control {
namespace osc {

/**
 * @brief A contact task that represents a rigid contact for a given frame
 * within the environment or an object
 *
 */
class ContactTask : public MotionTask {
 public:
  ContactTask() = default;
  ~ContactTask() = default;

  using SharedPtr = std::shared_ptr<ContactTask>;

  ContactTask(const std::string &name, const int &xdim, const int &vdim)
      : MotionTask(name, xdim, vdim) {}

  // Whether the point is in contact or not
  bool inContact = false;

  /**
   * @brief Surface normal for the contact surface
   *
   * @return Eigen::Vector3d
   */
  Eigen::Vector3d normal() { return normal_; }

  void SetNormal(const Eigen::Vector3d &normal) { normal_ = normal; }

  // Friction coefficient
  const double &mu() const { return mu_; }
  void SetFrictionCoefficient(const double &mu) { mu_ = mu; }

  /**
   * @brief Maximum allowable contact wrench
   *
   * @return Eigen::Ref<const Eigen::VectorXd>
   */
  virtual Eigen::Ref<const Eigen::VectorXd> fmin() = 0;

  /**
   * @brief Minimum allowable contact wrench
   *
   * @return Eigen::Ref<const Eigen::VectorXd>
   */
  virtual Eigen::Ref<const Eigen::VectorXd> fmax() = 0;

 private:
  // Friction coefficient
  double mu_ = 1.0;

  // Surface normal
  Eigen::Vector3d normal_;
};

class ContactTask3D : public ContactTask {
 public:
  ContactTask3D() = default;
  ~ContactTask3D() = default;

  ContactTask3D(const std::string &name, const casadi::SX &xpos,
                const casadi::SX &xvel, const casadi::SX &xacc)
      : ContactTask(name, 3, 3) {}

  struct Reference {
    Eigen::Vector3d x;
  };

  // TODO - Select frame that this reference is in

  const Reference &GetReference() { return ref_; }
  void SetReference(const Eigen::Vector3d &x) { ref_.x = x; }

  void ComputeMotionError() override;

  void SetFrictionForceLimits(const Eigen::Vector3d &fmin,
                              const Eigen::Vector3d &fmax) {
    fmin_ = fmin;
    fmax_ = fmax;
  }

  Eigen::Ref<const Eigen::VectorXd> fmin() override { return fmin_; }
  Eigen::Ref<const Eigen::VectorXd> fmax() override { return fmax_; }

 private:
  Reference ref_;
  Eigen::Vector3d fmax_;
  Eigen::Vector3d fmin_;
};

class ContactTask6D : public ContactTask {
 public:
  ContactTask6D() = default;
  ~ContactTask6D() = default;

  ContactTask6D(const std::string &name, const casadi::SX &xpos,
                const casadi::SX &xvel, const casadi::SX &xacc)
      : ContactTask(name, 6, 6) {}

  struct Reference {
    Eigen::Vector3d x;
    Eigen::Quaterniond q;
  };

  // TODO - Select frame that this reference is in

  const Reference &GetReference() { return ref_; }
  void SetReference(const Eigen::Vector3d &x) { ref_.x = x; }

  void ComputeMotionError() override;

  Eigen::Ref<const Eigen::VectorXd> fmin() override { return fmin_; }
  Eigen::Ref<const Eigen::VectorXd> fmax() override { return fmax_; }

 private:
  Reference ref_;
  Eigen::Vector<double, 6> fmax_;
  Eigen::Vector<double, 6> fmin_;
};

}  // namespace osc
}  // namespace control
}  // namespace damotion

#endif /* TASKS_CONTACT_H */
