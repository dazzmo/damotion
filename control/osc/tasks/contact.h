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

    ContactTask(const std::shared_ptr<TargetFrame> &frame) : Task() {
        Resize(3);
    }

    /**
     * @brief The TargetFrame the motion task is designed for
     *
     * @return const TargetFrame&
     */
    TargetFrame &Frame() { return *frame_; }

    // Desired pose translational component
    Eigen::Vector3d xr;

    // Whether the point is in contact or not
    bool inContact = false;

    // Contact surface normal
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();

    // Friction coefficient
    double mu = 1.0;

    // Maximum allowable force
    Eigen::Vector3d fmax = Eigen::Vector3d::Ones();
    // Minimum allowable force
    Eigen::Vector3d fmin = Eigen::Vector3d::Ones();

   private:
    std::shared_ptr<TargetFrame> frame_;
};

}
}
}


#endif/* TASKS_CONTACT_H */
