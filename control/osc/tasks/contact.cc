#include "control/osc/tasks/contact.h"

namespace damotion {
namespace control {
namespace osc {

void ContactTask3D::ComputeMotionError() {
    damotion::common::Profiler profiler("ContactTask3D::ComputeMotionError");

    e_ = pos() - GetReference().x;
    // Ensure no velocity
    de_ = vel();
}

void ContactTask6D::ComputeMotionError() {
    damotion::common::Profiler profiler("ContactTask6D::ComputeMotionError");

    throw std::runtime_error(
        "ContactTask6D::ComputeMotionError() is not currently implemented!");
}

}  // namespace osc
}  // namespace control
}  // namespace damotion
