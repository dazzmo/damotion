#ifndef TASKS_MOTION_H
#define TASKS_MOTION_H

#include "control/osc/tasks/task.h"

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

    MotionTask(const std::string &name,
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

   private:
    // Target frame the motion task is associated with
    std::shared_ptr<TargetFrame> frame_;
};

class PositionTask : public MotionTask {
   public:
    PositionTask() = default;
    ~PositionTask() = default;

    struct Reference {
        Eigen::Vector3d x;
        Eigen::Vector3d v;
    };

    PositionTask(const std::string &name,
                 const std::shared_ptr<TargetFrame> &frame)
        : MotionTask(name, frame) {
        Resize(3);
    }

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

    OrientationTask(const std::string &name,
                    const std::shared_ptr<TargetFrame> &frame)
        : MotionTask(name, frame) {
        Resize(3);
    }

    casadi::SX pos_sym() override {
        return Frame().pos_sym()(casadi::Slice(3, 7));
    }
    casadi::SX vel_sym() override {
        return Frame().vel_sym()(casadi::Slice(3, 6));
    }
    casadi::SX acc_sym() override {
        return Frame().acc_sym()(casadi::Slice(3, 6));
    }

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

class Pose6DTask : public MotionTask {
   public:
    Pose6DTask() = default;
    ~Pose6DTask() = default;

    struct Reference {
        Eigen::Vector3d x;
        Eigen::Quaterniond q;
        Eigen::Vector3d v;
        Eigen::Vector3d w;
    };

    Pose6DTask(const std::string &name,
               const std::shared_ptr<TargetFrame> &frame)
        : MotionTask(name, frame) {
        Resize(6);
    }

    casadi::SX pos_sym() override { return Frame().pos_sym(); }
    casadi::SX vel_sym() override { return Frame().vel_sym(); }
    casadi::SX acc_sym() override { return Frame().acc_sym(); }

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
