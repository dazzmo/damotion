#ifndef PIECEWISE_BEZIER_H
#define PIECEWISE_BEZIER_H

#include "common/trajectory/bezier.h"

namespace damotion {
namespace trajectory {

template <typename T>
class PiecewiseBezier {
 public:
  PiecewiseBezier() : sz_(0), duration_(0) { trajectories_.reserve(100); }

  Bezier<T>& trajectory(int i) { return trajectories_[i]; }

  const int& size() const { return sz_; }

  void AppendTrajectory(Bezier<T>& trajectory) {
    Bezier<T> tmp = trajectory;
    tmp.OffsetTrajectory(duration_);
    trajectories_.push_back(tmp);
    duration_ += tmp.duration();
    sz_++;
  }

 private:
  int sz_;
  std::vector<Bezier<T>> trajectories_;
  T duration_;
};

}  // namespace trajectory
}  // namespace damotion
#endif /* PIECEWISE_BEZIER_H */
