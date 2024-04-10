#ifndef TASKS_TASK_H
#define TASKS_TASK_H

#include <Eigen/Core>
#include <algorithm>
#include <casadi/casadi.hpp>
#include <map>
#include <string>

#include "common/profiler.h"
#include "solvers/constraint.h"
#include "solvers/cost.h"
#include "solvers/program.h"
#include "utils/casadi.h"
#include "utils/codegen.h"
#include "utils/eigen_wrapper.h"
#include "utils/pinocchio_model.h"

namespace opt = damotion::optimisation;
namespace sym = damotion::symbolic;
namespace utils = damotion::utils;

namespace damotion {
namespace control {
namespace osc {

typedef utils::casadi::PinocchioModelWrapper::TargetFrame TargetFrame;

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
    const int dim() const { return dim_; }

    const std::string & name() const { return name_; }


    /**
     * @brief Add a parameter p along with its program reference for the given
     * task
     *
     * @param p
     */
    void AddParameter(const casadi::SX &p,
                      Eigen::Ref<const Eigen::MatrixXd> &pref) {
        ps_.push_back(p);
        pv_.push_back(pref);
    }

    casadi::SXVector &SymbolicParameters() { return ps_; }
    sym::ParameterRefVector &Parameters() { return pv_; }

    /**
     * @brief Resizes the dimension of the task.
     *
     * @param ndim
     */
    void Resize(const int ndim) {
        dim_ = ndim;
        e_ = Eigen::VectorXd::Zero(ndim);
        de_ = Eigen::VectorXd::Zero(ndim);
        Kp_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(ndim);
        Kp_.setZero();
        Kd_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(ndim);
        Kd_.setZero();
    }

    const Eigen::VectorXd &Error() { return e_; }
    const Eigen::VectorXd &ErrorDerivative() { return de_; }

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

    casadi::SXVector ps_;
    sym::ParameterRefVector pv_;

   private:
    int dim_ = 0;
    std::string name_;
};

}
}
}


#endif/* TASKS_TASK_H */
