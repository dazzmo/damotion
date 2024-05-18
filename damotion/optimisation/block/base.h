#ifndef BLOCK_BASE_H
#define BLOCK_BASE_H

/**
 * @file block.h
 * @author Damian Abood (damian.abood@sydney.edu.au)
 * @brief Block matrix class to handle assemblage of large matrices and vectors
 * such as gradients, Jacobians and Hessians from bindings for use in numerical
 * optimisation.
 * @version 0.1
 * @date 2024-05-15
 *
 */
#ifndef OPTIMISATION_BLOCK_H
#define OPTIMISATION_BLOCK_H

#include <Eigen/Sparse>
#include <unordered_map>
#include <vector>

#include "damotion/optimisation/binding.h"
#include "damotion/optimisation/fwd.h"
#include "damotion/optimisation/program.h"

namespace damotion {
namespace optimisation {

class BlockMatrixFunctionBase {
 public:
  enum class Type {
    kVector = 0,
    kJacobian,
    kHessian,
    kGradient,
    kQuadratic,
    kLinear
  };

  BlockMatrixFunctionBase(const int &rows, const int &cols, const Type &type,
                          DecisionVariableManager &x_manager)
      : rows_(rows), cols_(cols), nc_(0), type_(type), x_manager_(x_manager) {}

  const Type &GetType() const { return type_; }

  /**
   * @brief Number of rows of the block matrix
   *
   * @return const int&
   */
  const int &rows() const { return rows_; }

  /**
   * @brief Number of cols of the block matrix
   *
   * @return const int&
   */
  const int &cols() const { return cols_; }

 protected:
  // Number of constraints added (if using Jacobian)
  int nc_;
  // Decision variable manager which handles variable indices
  DecisionVariableManager &x_manager_;

 private:
  int rows_;
  int cols_;

  Type type_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* OPTIMISATION_BLOCK_H */

#endif /* BLOCK_BASE_H */
