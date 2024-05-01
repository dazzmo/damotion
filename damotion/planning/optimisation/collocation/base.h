#ifndef COLLOCATION_BASE_H
#define COLLOCATION_BASE_H

namespace damotion {
namespace planning {
namespace optimisation {

class CollocationConstraintBase {
 public:
 private:
  int nx_ = 0;
  int nu_ = 0;
};

class TrapezoidalCollocationConstraint {};

class HermiteSimpsonCollocationConstraint {};

}  // namespace optimisation
}  // namespace planning
}  // namespace damotion

#endif /* COLLOCATION_BASE_H */
