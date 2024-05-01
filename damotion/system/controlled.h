#ifndef SYSTEM_CONTROLLED_H
#define SYSTEM_CONTROLLED_H

#include "system/system.h"
#include "utils/pinocchio_model.h"

namespace damotion {

namespace system {

class ControlledSystem : public System {
 public:
  /**
   * @brief Empty constructor
   *
   */
  ControlledSystem() : System(), nu_(0) {}

  /**
   * @brief Construct a new Controlled System object with state and state
   * derivative dimension nx and input dimension nu.
   *
   * @param nx Dimension of state and state derivative
   * @param nu Dimension of input
   */
  ControlledSystem(int nx, int nu) : System(nx), nu_(nu) {}

  /**
   * @brief Construct a new Controlled System object with state dimension nx,
   * state derivative dimension ndx and input dimension nu.
   *
   * @param nx Dimension of state
   * @param ndx Dimension of state derivative
   * @param nu Dimension of input
   */
  ControlledSystem(int nx, int ndx, int nu) : System(nx, ndx), nu_(nu) {}

  ~ControlledSystem() = default;

  /**
   * @brief Dimension of the input
   *
   * @return const int
   */
  const int nu() const { return nu_; }

  /**
   * @brief Forward dynamics of the system, should be of the form $\f \dot{x}
   * = f(x, u) \f$
   *
   */
  casadi::Function dynamics() { return System::dynamics(); }

  /**
   * @brief Set the function to evaluate the forward dynamics of the system.
   * Function should be of the form $\f \dot{x}= f(x, u) \f$
   *
   *
   * @param f
   */
  void setDynamics(const casadi::Function &f) { fid_ = f; }

  /**
   * @brief Inverse dynamics of the system, should be of the form $\f \tau =
   * f(x, \dot{x}) \f$
   *
   * @return casadi::Function
   */
  casadi::Function inverseDynamics() { return fid_; }

  /**
   * @brief Set the function to evaluate the inverse dynamics of the system.
   * Function should be of the form $\f \tau = f(x, \dot{x}) \f$
   *
   *
   * @param f
   */
  void setInverseDynamics(const casadi::Function &f) { fid_ = f; }

 protected:
 private:
  // Dimension of input
  int nu_;

  // Function to compute inverse dynamics
  casadi::Function fid_;
};

/**
 * @brief Controlled second-order dynamic system
 *
 */
class SecondOrderControlledSystem : public ControlledSystem {
 public:
  /**
   * @brief Empty constructor
   *
   */
  SecondOrderControlledSystem() : ControlledSystem() {}

  /**
   * @brief Construct a new Controlled System object with configuration and
   * tangent space dimension nq and input dimension nu.
   *
   * @param nq Dimension of configuration and tangent space
   * @param nu Dimension of input
   */
  SecondOrderControlledSystem(int nq, int nu) : ControlledSystem(nq + nq, nu) {}

  /**
   * @brief Construct a new Controlled System object with state dimension nx,
   * state derivative dimension ndx and input dimension nu.
   *
   * @param nq Dimension of configuration
   * @param nv Dimension of tangent space
   * @param nu Dimension of input
   */
  SecondOrderControlledSystem(int nq, int nv, int nu)
      : ControlledSystem(nq, nv, nu) {}

  SecondOrderControlledSystem(utils::casadi::PinocchioModelWrapper &wrapper) {
    *this = wrapper;
  }

  ~SecondOrderControlledSystem() = default;

  SecondOrderControlledSystem &operator=(
      utils::casadi::PinocchioModelWrapper wrapper);

  /**
   * @brief Dimension of the configuration of the system
   *
   * @return const int
   */
  const int nq() const { return nq_; }

  /**
   * @brief Dimension of the tangent space of the system
   *
   * @return const int
   */
  const int nv() const { return nv_; }

  /**
   * @brief Forward dynamics of the system, should be of the form $\f \dot{x}
   * = f(x, u) \f$ with $\f x = [q, \dot{q}]^T \f$.
   *
   */
  casadi::Function dynamics() { return ControlledSystem::dynamics(); }

  /**
   * @brief Inverse dynamics of the system, should be of the form $\f \tau =
   * f(q, \dot{q}, \ddot{q}) \f$
   *
   * @return casadi::Function
   */
  casadi::Function inverseDynamics() {
    return ControlledSystem::inverseDynamics();
  }

  /**
   * @brief Set the function to evaluate the inverse dynamics of the system.
   * Function should be of the form $\f \tau = f(q, \dot{q}, \ddot{q}) \f$
   *
   * @param f Inverse dynamics function
   */
  void setInverseDynamics(const casadi::Function &f) {
    ControlledSystem::setInverseDynamics(f);
  }

 protected:
 private:
  // Dimension of configuration space
  int nq_;
  // Dimension of tangent space
  int nv_;
};

}  // namespace system
}  // namespace damotion

#endif /* SYSTEM_CONTROLLED_H */
