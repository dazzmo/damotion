#ifndef TASKS_CONTACT_H
#define TASKS_CONTACT_H

#include "control/osc/tasks/task.h"

namespace opt = damotion::optimisation;
namespace sym = damotion::symbolic;
namespace utils = damotion::utils;

namespace damotion {
namespace control {
namespace osc {

/**
 * @brief A contact task that represents a rigid contact for a given frame
 * within the environment or an object
 *
 */
class ContactTask : public Task {
   public:
    ContactTask() = default;
    ~ContactTask() = default;

    ContactTask(const std::string &name,
                const std::shared_ptr<TargetFrame> &frame)
        : Task(name), frame_(frame) {}

    /**
     * @brief Symbolic position of the frame the motion task is based on
     *
     * @return const casadi::SX
     */
    virtual casadi::SX pos_sym() = 0;

    /**
     * @brief Symbolic velocity of the frame the motion task is based on
     *
     * @return casadi::SX
     */
    virtual casadi::SX vel_sym() = 0;

    /**
     * @brief Symbolic acceleration of the frame the motion task is based on
     *
     * @return casadi::SX
     */
    virtual casadi::SX acc_sym() = 0;

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
     * @brief The TargetFrame the motion task is designed for
     *
     * @return const TargetFrame&
     */
    TargetFrame &Frame() { return *frame_; }

    /**
     * @brief Compute the desired tracking task acceleration as a PD error
     * metric on the task position and velocity errors
     *
     * @return Eigen::VectorXd
     */
    virtual void ComputeMotionError() = 0;

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
     * @return Eigen::VectorXd
     */
    virtual Eigen::VectorXd fmin() = 0;

    /**
     * @brief Minimum allowable contact wrench
     *
     * @return Eigen::VectorXd
     */
    virtual Eigen::VectorXd fmax() = 0;

   private:
    // Friction coefficient
    double mu_ = 1.0;

    // Surface normal
    Eigen::Vector3d normal_;

    // Target frame the motion task is associated with
    std::shared_ptr<TargetFrame> frame_;
};

class ContactTask3D : public ContactTask {
   public:
    ContactTask3D() = default;
    ~ContactTask3D() = default;

    ContactTask3D(const std::string &name,
                  const std::shared_ptr<TargetFrame> &frame)
        : ContactTask(name, frame) {
        Resize(3);  // TODO Rename this to ResizeTask()
    }

    struct Reference {
        Eigen::Vector3d x;
    };

    // TODO - Select frame that this reference is in

    const Reference &GetReference() { return ref_; }
    void SetReference(const Eigen::Vector3d &x) { ref_.x = x; }

    casadi::SX pos_sym() override {
        return Frame().pos_sym()(casadi::Slice(0, 3));
    }
    casadi::SX vel_sym() override {
        return Frame().vel_sym()(casadi::Slice(0, 3));
    }
    casadi::SX acc_sym() override {
        return Frame().acc_sym()(casadi::Slice(0, 3));
    }

    Eigen::VectorXd pos() override { return Frame().pos().topRows(3); }
    Eigen::VectorXd vel() override { return Frame().vel().topRows(3); }
    Eigen::VectorXd acc() override { return Frame().acc().topRows(3); }

    void ComputeMotionError() override;

    void SetFrictionForceLimits(const Eigen::Vector3d &fmin,
                                const Eigen::Vector3d &fmax) {
        fmin_ = fmin;
        fmax_ = fmax;
    }

    Eigen::VectorXd fmin() override { return fmin_; }
    Eigen::VectorXd fmax() override { return fmax_; }

   private:
    Reference ref_;
    Eigen::Vector3d fmax_;
    Eigen::Vector3d fmin_;
};

class ContactTask6D : public ContactTask {
   public:
    ContactTask6D() = default;
    ~ContactTask6D() = default;

    ContactTask6D(const std::string &name,
                  const std::shared_ptr<TargetFrame> &frame)
        : ContactTask(name, frame) {
        Resize(6);  // TODO Rename this to ResizeTask()
    }

    struct Reference {
        Eigen::Vector3d x;
        Eigen::Quaterniond q;
    };

    // TODO - Select frame that this reference is in

    const Reference &GetReference() { return ref_; }
    void SetReference(const Eigen::Vector3d &x) { ref_.x = x; }

    casadi::SX pos_sym() override { return Frame().pos_sym(); }
    casadi::SX vel_sym() override { return Frame().vel_sym(); }
    casadi::SX acc_sym() override { return Frame().acc_sym(); }

    Eigen::VectorXd pos() override { return Frame().pos(); }
    Eigen::VectorXd vel() override { return Frame().vel(); }
    Eigen::VectorXd acc() override { return Frame().acc(); }

    void ComputeMotionError() override;

    Eigen::VectorXd fmin() override { return fmin_; }
    Eigen::VectorXd fmax() override { return fmax_; }

   private:
    Reference ref_;
    Eigen::Vector<double, 6> fmax_;
    Eigen::Vector<double, 6> fmin_;
};

}  // namespace osc
}  // namespace control
}  // namespace damotion

#endif /* TASKS_CONTACT_H */
