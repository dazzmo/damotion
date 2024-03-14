#ifndef TRAJECTORY_TRAJECTORY_H
#define TRAJECTORY_TRAJECTORY_H

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <iostream>
#include <vector>

namespace damotion {
namespace trajectory {

/**
 * @brief The trajectory class represents the motion of a system over some
 * duration of time. It requires the ability to evaluate the trajectory along
 * the path, as well as derivatives along the trajectory
 *
 * @tparam T
 */
template <typename Scalar>
class TrajectoryBase {
   public:
    /**
     * @brief Construct a new Trajectory Base object with dimension dim with
     * start and ending times set to zero
     *
     * @param dim Dimension of the trajectory
     */
    TrajectoryBase(const int& dim)
        : dim_(dim), t0_(Scalar(0)), tf_(Scalar(0)) {}
    ~TrajectoryBase() = default;

    /**
     * @brief Returns the dimension of the trajectory \f$ x(t) \f$
     *
     * @return const int&
     */
    const int& dim() const { return dim_; }

    /**
     * @brief Initial time \f$ t_0 \f$ that the trajectory \f$ x(t) \f$ begins
     * at
     *
     * @return const T&
     */
    const Scalar& t0() const { return t0_; }

    /**
     * @brief Final time \f$ t_f \f$ that the trajectory \f$ x(t) \f$ ends at
     *
     * @return const T&
     */
    const Scalar& tf() const { return tf_; }

    /**
     * @brief Update the starting time of the trajectory
     *
     * @param t The time to set the start to
     */
    void UpdateBeginningTime(const Scalar& t) { t0_ = t; }

    /**
     * @brief The total duration of the trajectory
     *
     * @return T
     */
    Scalar duration() {
        duration_ = tf_ - t0_;
        return duration_;
    }

    /**
     * @brief Offsets the trajectory in its dependent variable by
     * some value offset such that the starting time is increased by offset.
     *
     * @param offset
     */
    void OffsetTrajectory(const Scalar& offset) {
        t0_ += offset;
        tf_ += offset;
    }

   protected:
    // Dimension of the trajectory
    int dim_;
    // Starting time of the trajectory
    Scalar t0_;
    // Ending time of the trajectory
    Scalar tf_;
    // Duration of the trajectory
    Scalar duration_;
};

/**
 * @brief Continuous trajectory class to represent the profiles x(t)
 *
 * @tparam Scalar
 */
template <typename Scalar>
class Trajectory : public TrajectoryBase<Scalar> {
   public:
    /**
     * @brief Construct a new Trajectory object with dimension dim
     *
     * @param dim Dimension of the trajectory
     */
    Trajectory(const int& dim) : TrajectoryBase<Scalar>(dim) {}

    virtual Eigen::VectorX<Scalar> eval(const Scalar& t) = 0;
    virtual Eigen::VectorX<Scalar> derivative(const Scalar& t,
                                              const int order) = 0;

    /**
     * @brief Update the ending time of the trajectory
     *
     * @param t
     */
    void UpdateEndingTime(const Scalar& t) { this->tf_ = t; }

    /**
     * @brief Trajectory value at the starting position given by t0()
     * @param order Order of derivative to return
     * @return Eigen::VectorX<T>
     */
    Eigen::VectorX<Scalar> begin(const int order = 0) {
        if (order == 0) {
            return eval(this->t0_);
        } else {
            return derivative(this->t0_, order);
        }
    }

    /**
     * @brief Trajectory value at the ending position given by tf()
     *
     * @param order Order of derivative to return
     * @return  Eigen::VectorX<T>
     */
    Eigen::VectorX<Scalar> end(const int order = 0) {
        if (order == 0) {
            return eval(this->tf_);
        } else {
            return derivative(this->tf_, order);
        }
    }

   protected:
};

/**
 * @brief Determines if the trajectory is contained within the limits xl and xu
 * such that xl < x(t) < xu. This is only evaluated at a finite number of
 * points, with this value determined by resolution
 *
 * @param trajectory
 * @param xl
 * @param xu
 * @param resolution
 * @return true
 * @return false
 */
bool IsBoundedWithinBoxLimits(Trajectory<double>& trajectory,
                              const Eigen::VectorXd& xl, const Eigen::VectorXd& xu,
                              double resolution = 10);

/**
 * @brief Creates a discrete trajectory from vectors x and t indicating the
 * values of the state and time respectively
 *
 * @tparam Scalar The scalar type used
 */
template <typename Scalar>
class DiscreteTrajectory : public TrajectoryBase<Scalar> {
   public:
    DiscreteTrajectory(const std::vector<Eigen::VectorX<Scalar>>& x,
                       const std::vector<Scalar>& t) : TrajectoryBase<Scalar>(x[0].size()) {
        x_ = x;
        t_ = t;
    }
    ~DiscreteTrajectory() = default;

    /**
     * @brief The vector of state values that comprise the discrete trajectory
     *
     * @return const std::vector<Eigen::VectorXd>&
     */
    const std::vector<Eigen::VectorXd>& x() const { return x_; }

    /**
     * @brief The vector of times for the discrete trajectory
     *
     * @return const std::vector<double>&
     */
    const std::vector<double>& t() const { return t_; }

    /**
     * @brief Evaluate the trajectory at the current index idx (i.e. returns \f$
     * x_{idx} \f$)
     *
     * @param idx
     * @return const MatrixType&
     */
    const Eigen::VectorX<Scalar>& eval(const int& idx) const {
        return x_[idx];
    }

    const Eigen::VectorX<Scalar>& eval(Scalar& t,
                                       const Scalar& precision = 0.01) const;

    /**
     * @brief Trajectory value at the starting position given by t0()
     *
     * @return const Eigen::VectorX<Scalar>&
     */
    const Eigen::VectorX<Scalar>& begin() const { return x_.begin(); }

    /**
     * @brief Trajectory value at the starting position given by tf()
     *
     * @return const Eigen::VectorX<Scalar>&
     */
    const Eigen::VectorX<Scalar>& end() const { return x_.end(); }

    // Vector of durations for each discrete value in the trajectory
    std::vector<Scalar>& t() { return t_; }
    
   private:
    // Vector of trajectory values
    std::vector<Eigen::VectorX<Scalar>> x_;
    // Vector of times
    std::vector<Scalar> t_;
};

// Template specialisation
template <>
const Eigen::VectorX<double>& DiscreteTrajectory<double>::eval(
    double& t, const double& precision) const;

template <typename Scalar>
class PiecewiseDiscreteTrajectory {
   public:
    PiecewiseDiscreteTrajectory() : sz_(0) { trajectories_.reserve(100); }

    DiscreteTrajectory<Scalar>& trajectory(int i) { return trajectories_[i]; }

    const int& size() const { return sz_; }

    void AppendTrajectory(DiscreteTrajectory<Scalar>& trajectory) {
        DiscreteTrajectory<Scalar> tmp = trajectory;
        tmp.OffsetTrajectory(duration_);
        trajectories_.push_back(tmp);
        duration_ += tmp.duration();
        sz_++;
    }

    /**
     * @brief Compresses the piecewise discrete trajectory into a single
     * discrete trajectory
     *
     * @return DiscreteTrajectory<Scalar>
     */
    DiscreteTrajectory<Scalar> ToSingleTrajectory() {
        std::vector<Eigen::VectorX<Scalar>> x;
        std::vector<Scalar> t;

        // Determine total size of the trajectory
        int sz_out = 0;
        for (int i = 0; i < sz_; ++i) {
            sz_out += trajectory(i).x().size();
        }

        x.reserve(sz_out);
        t.reserve(sz_out);

        // Create new discrete trajectory
        int cnt = 0;
        Scalar t_offset = Scalar(0);
        for (int i = 0; i < sz_; ++i) {
            for (int j = 0; j < trajectory(i).x().size(); ++j) {
                x.push_back(trajectory(i).x()[j]);
                t.push_back(t_offset + trajectory(i).t()[j]);
            }
            t_offset = t.back();
        }

        return DiscreteTrajectory<Scalar>(x, t);
    }

#ifdef DAMOTION_WITH_PYBIND11

#endif

   private:
    int sz_;
    std::vector<DiscreteTrajectory<Scalar>> trajectories_;
    Scalar duration_;
};

}  // namespace trajectory
}  // namespace damotion

#endif /* TRAJECTORY_TRAJECTORY_H */
