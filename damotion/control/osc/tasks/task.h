#ifndef TASKS_TASK_H
#define TASKS_TASK_H

#include <Eigen/Core>
#include <algorithm>
#include <casadi/casadi.hpp>
#include <map>
#include <string>

#include "damotion/casadi/codegen.h"
#include "damotion/casadi/eigen.h"
#include "damotion/casadi/function.h"
#include "damotion/casadi/pinocchio_model.h"
#include "damotion/common/profiler.h"
#include "damotion/control/fwd.h"
#include "damotion/optimisation/constraints/constraints.h"
#include "damotion/optimisation/costs/costs.h"
#include "damotion/optimisation/program.h"

namespace damotion {
namespace control {
namespace osc {

class Task {
 public:
  Task() = default;
  ~Task() = default;

  using SharedPtr = std::shared_ptr<Task>;

  /**
   * @brief Construct a new Task object with name, configuration space of size
   * xdim and tangent space of size vdim.
   *
   * @param name Name of the task
   * @param xdim dimension of the configuration space of the task
   * @param vdim dimension of the tangent space of the task
   */
  Task(const std::string &name, const int &xdim, const int &vdim)
      : name_(name) {
    ResizeTask(xdim, vdim);
  }

  /**
   * @brief dimension of the task configuration space.
   *
   * @return const int
   */
  const int &xdim() const { return xdim_; }

  /**
   * @brief dimension of the task tangent space
   *
   * @return const int&
   */
  const int &vdim() const { return vdim_; }

  const std::string &name() const { return name_; }

  /**
   * @brief Position of the task (xdim x 1)
   *
   * @return Eigen::VectorXd
   */
  Eigen::VectorXd pos() { return f_->getOutput(0); };

  /**
   * @brief Velocity of the task (vdim x 1)
   *
   * @return Eigen::VectorXd
   */
  Eigen::VectorXd vel() { return f_->getOutput(1); }

  /**
   * @brief Acceleration of the task (vdim x 1)
   *
   * @return Eigen::VectorXd
   */
  Eigen::VectorXd acc() { return f_->getOutput(2); }

  /**
   * @brief Resizes the dimension of the task.
   *
   * @param xdim dimension of the task configuration space
   * @param vdim dimension of the task tangent space
   */
  void ResizeTask(const int &xdim, const int &vdim) {
    // Set dimension of task
    xdim_ = xdim;
    vdim_ = vdim;
    // Weighting
    w_ = Eigen::VectorXd::Ones(vdim);
    // Task error
    e_ = Eigen::VectorXd::Zero(vdim);
    // Task error derivative
    de_ = Eigen::VectorXd::Zero(vdim);
    // Task PD proportional gain
    Kp_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(vdim);
    Kp_.setZero();
    // Task PD derivative gain
    Kd_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(vdim);
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

  void UpdateState(const Eigen::VectorXd &qpos, const Eigen::VectorXd &qvel,
                   const Eigen::VectorXd &qacc) {
    f_->call({qpos, qvel, qacc});
  }

  /**
   * @brief Set function to compute the task position, velocity and
   * acceleration.
   *
   * @param f
   */
  void SetFunction(const common::Function<Eigen::VectorXd>::SharedPtr &f) {
    f_ = f;
  }

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
  // dimension of the task configurations space
  int xdim_ = 0;
  // dimension of the task tangent space
  int vdim_ = 0;
  std::string name_;

  // Wrapper for the function of the symbolic function
  common::Function<Eigen::VectorXd>::SharedPtr f_;
};

}  // namespace osc
}  // namespace control
}  // namespace damotion

#endif /* TASKS_TASK_H */
