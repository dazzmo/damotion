#ifndef SYSTEM_CONTROLLED_H
#define SYSTEM_CONTROLLED_H

#include "system/system.h"

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
     * = f(x, u, t) \f$
     *
     */
    casadi::Function dynamics() { return System::dynamics(); }

    /**
     * @brief Inverse dynamics of the system, should be of the form $\f \tau =
     * f(x, \dot{x}, t) \f$
     *
     * @return casadi::Function
     */
    casadi::Function inverseDynamics() { return fid_; }

   protected:
    virtual casadi::Function inverseDynamicsImpl() = 0;

    void setInverseDynamics(casadi::Function &f) { fid_ = f; }

   private:
    // Dimension of input
    int nu_;

    // Function to compute inverse dynamics
    casadi::Function fid_;
};

#endif /* SYSTEM_CONTROLLED_H */
