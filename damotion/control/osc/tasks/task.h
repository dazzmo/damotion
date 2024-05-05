#ifndef TASKS_TASK_H
#define TASKS_TASK_H

#include <Eigen/Core>
#include <algorithm>
#include <casadi/casadi.hpp>
#include <map>
#include <string>

#include "damotion/common/profiler.h"
#include "damotion/model/frame.h"
#include "damotion/optimisation/constraints/constraints.h"
#include "damotion/optimisation/costs/costs.h"
#include "damotion/optimisation/program.h"
#include "damotion/utils/codegen.h"
#include "damotion/utils/eigen_wrapper.h"
#include "damotion/utils/pinocchio_model.h"

namespace opt = damotion::optimisation;
namespace sym = damotion::symbolic;
namespace utils = damotion::utils;

namespace damotion {
namespace control {
namespace osc {

class Task {
 public:
  Task() = default;
  ~Task() = default;

  Task(const std::string &name) : name_(name) {}

  /**
   * @brief Dimension of the task.
   *
   * @return const int
   */
  const int &dim() const { return dim_; }

  const std::string &name() const { return name_; }

  /**
   * @brief Resizes the dimension of the task.
   *
   * @param ndim
   */
  void ResizeTask(const int ndim) {
    // Set dimension of task
    dim_ = ndim;
    // Weighting
    w_ = Eigen::VectorXd::Ones(ndim);
    // Task error
    e_ = Eigen::VectorXd::Zero(ndim);
    // Task error derivative
    de_ = Eigen::VectorXd::Zero(ndim);
    // Task PD proportional gain
    Kp_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(ndim);
    Kp_.setZero();
    // Task PD derivative gain
    Kd_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(ndim);
    Kd_.setZero();
  }

  /**
   * @brief Returns the current error in the task within the task-space
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &Error() { return e_; }

  /**
   * @brief Returns the current derivative error in the task within the
   * task-space
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &ErrorDerivative() { return de_; }

  /**
   * @brief The current weights for each dimension of the given task
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &Weighting() { return w_; }

  /**
   * @brief Set the weighting of the task to the vector w
   *
   * @param w
   */
  Task &SetWeighting(const Eigen::VectorXd &w) {
    w_ = w;
    return *this;
  }

  /**
   * @brief Sets all weightings of the task to constant value w
   *
   * @param w
   */
  Task &SetWeighting(const double &w) {
    w_.setConstant(w);
    return *this;
  }

  Task &SetKpGains(const Eigen::VectorXd &Kp) {
    Kp_.diagonal() = Kp;
    return *this;
  }

  Task &SetKdGains(const Eigen::VectorXd &Kd) {
    Kd_.diagonal() = Kd;
    return *this;
  }

  virtual Eigen::VectorXd GetPDError() { return Kp_ * e_ + Kd_ * de_; }

 protected:
  // Task error
  Eigen::VectorXd e_;
  Eigen::VectorXd de_;
  // PD Tracking gains
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kp_;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kd_;
  // Task weighting
  Eigen::VectorXd w_;

 private:
  int dim_ = 0;
  std::string name_;
};

}  // namespace osc
}  // namespace control
}  // namespace damotion

#endif /* TASKS_TASK_H */
