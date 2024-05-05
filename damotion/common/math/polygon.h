#ifndef MATH_POLYGON_H
#define MATH_POLYGON_H

#include <Eigen/Core>
#include <vector>

namespace damotion {
namespace common {

class Polygon {
 public:
  void AddPoint(const Eigen::VectorXd &p) {}

  // Returns a series of inequality constraints that indicate a point is
  // contained within the polygon

 private:
  std::vector<Eigen::VectorXd> points_;
};

}  // namespace common
}  // namespace damotion

#endif /* MATH_POLYGON_H */
