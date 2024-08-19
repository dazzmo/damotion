#ifndef CONSTRAINTS_BOUNDING_BOX_H
#define CONSTRAINTS_BOUNDING_BOX_H

#include "damotion/optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

// class BoundingBoxConstraint : public Constraint {
//  public:
//   BoundingBoxConstraint(const std::string &name) {}

//   /**
//    * @brief Provides the bounds for the bounding box constraint. On default, it
//    * returns the bounds set with setLowerBound() and setUpperBound()
//    * respectively.
//    *
//    * @param xl
//    * @param xu
//    */
//   virtual void bounds(OptionalVector xl = nullptr,
//                       OptionalVector xu = nullptr) {
//     if (xl) *xl = this->lb();
//     if (xu) *xu = this->ub();
//   }

//   ReturnType eval(const InputVectorType &x, OptionalJacobianType J = nullptr) {
//     // Perform evaluation depending on what method is used
//     bounds(xl_, xu_);
//     if (J) {
//       // J->topRows().setIdentity();
//       // J->bottomRows().setIdentity();
//     }
//     return x - xl_;  // todo (damian) - add xu and xl as single vector
//   }

//  private:
//   Eigen::VectorXd xl_;
//   Eigen::VectorXd xu_;
// };

}  // namespace optimisation
}  // namespace damotion

#endif/* CONSTRAINTS_BOUNDING_BOX_H */
