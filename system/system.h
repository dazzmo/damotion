#ifndef SYSTEM_SYSTEM_H
#define SYSTEM_SYSTEM_H

#include <casadi/casadi.hpp>

/**
 * @brief Generic system representation of the form $\f \dot{x} = f(x, t) \f$
 *
 */
class System {
   public:
    /**
     * @brief Empty constructor
     *
     */
    System() : nx_(0), ndx_(0) {}

    /**
     * @brief Construct a new System object. Sets both state and state
     * derivative dimension to nx
     *
     * @param nx Dimension of the state and derivative state of the system
     */
    System(int nx) : nx_(nx), ndx_(nx) {}

    /**
     * @brief Construct a new System object.
     *
     * @param nx Dimension of the state of the system
     * @param ndx Dimension of the derivative of the state of the system
     */
    System(int nx, int ndx) : nx_(nx), ndx_(ndx) {}

    ~System() = default;

    /**
     * @brief Dimension of the state
     *
     * @return const int
     */
    const int nx() const { return nx_; }

    /**
     * @brief Dimension of the state derivative
     *
     * @return const int
     */
    const int ndx() const { return ndx_; }

    /**
     * @brief Forward dynamics of the system, should be of the form $\f \dot{x}
     * = f(x, t) \f$
     *
     * @return casadi::Function
     */
    casadi::Function dynamics() { return fd_; };

    
    void setDynamics(const casadi::Function &f) { fd_ = f; }

   protected:
   private:
    // Dimension of state
    int nx_;
    // Dimension of state derivative
    int ndx_;

    // Dynamics function
    casadi::Function fd_;
};

#endif /* SYSTEM_SYSTEM_H */
