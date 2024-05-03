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
#include "damotion/utils/casadi.h"
#include "damotion/utils/codegen.h"
#include "damotion/utils/eigen_wrapper.h"
#include "damotion/utils/pinocchio_model.h"

namespace opt = damotion::optimisation;
namespace sym = damotion::symbolic;
namespace utils = damotion::utils;

namespace damotion {
namespace control {
namespace osc {

typedef model::TargetFrame TargetFrame;

class Task {
 public:
  Task() = default;
  ~Task() = default;

  Task(const std::string &name) : name_(name) {}

  /**
   * @brief References for a motion
   *
   */
  struct Reference {
    Eigen::Vector3d xr;
    Eigen::Quaterniond qr;

    Eigen::Vector3d vr;
    Eigen::Vector3d wr;
  };

  /**
   * @brief Dimension of the task.
   *
   * @return const int
   */
  const int &dim() const { return dim_; }

  const std::string &name() const { return name_; }

  /**
   * @brief Add a parameter p along with its program reference for the given
   * task
   *
   * @param p
   */
  void AddParameter(const casadi::SX &p, const sym::Parameter &par) {
    ps_.push_back(p);
    pv_.push_back(par);
  }

  casadi::SXVector &SymbolicParameters() { return ps_; }
  sym::ParameterVector &Parameters() { return pv_; }

  /**
   * @brief Resizes the dimension of the task.
   *
   * @param ndim
   */
  void ResizeTask(const int ndim) {
    dim_ = ndim;
    w_ = Eigen::VectorXd::Ones(ndim);
    e_ = Eigen::VectorXd::Zero(ndim);
    de_ = Eigen::VectorXd::Zero(ndim);
    Kp_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(ndim);
    Kp_.setZero();
    Kd_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(ndim);
    Kd_.setZero();
  }

  const Eigen::VectorXd &Error() { return e_; }
  const Eigen::VectorXd &ErrorDerivative() { return de_; }

  const Eigen::VectorXd &Weighting() { return w_; }
  /**
   * @brief Set the weighting of the task to the vector w
   *
   * @param w
   */
  void SetWeighting(const Eigen::VectorXd &w) { w_ = w; }
  /**
   * @brief Sets all weightings of the task to w
   *
   * @param w
   */
  void SetWeighting(const double &w) { w_.setConstant(w); }

  void SetKpGains(const Eigen::VectorXd &Kp) { Kp_.diagonal() = Kp; }
  void SetKdGains(const Eigen::VectorXd &Kd) { Kd_.diagonal() = Kd; }

  virtual Eigen::VectorXd GetPDError() { return Kp_ * e_ + Kd_ * de_; }

 protected:
  // Task error
  Eigen::VectorXd e_;
  Eigen::VectorXd de_;
  // PD Tracking gains
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kp_;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kd_;

  // Task parameters
  casadi::SXVector ps_;
  sym::ParameterVector pv_;

 private:
  int dim_ = 0;
  std::string name_;
  Eigen::VectorXd w_;
};

}  // namespace osc
}  // namespace control
}  // namespace damotion

#endif /* TASKS_TASK_H */
