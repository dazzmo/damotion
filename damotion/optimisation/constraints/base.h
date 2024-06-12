#ifndef CONSTRAINTS_BASE_H
#define CONSTRAINTS_BASE_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>

#include "damotion/casadi/eigen.h"
#include "damotion/casadi/function.h"
#include "damotion/common/eigen.h"
#include "damotion/optimisation/bounds.h"
#include "damotion/optimisation/fwd.h"

namespace damotion {
namespace optimisation {

class Constraint {
 public:
  Constraint() = default;
  ~Constraint() = default;

  /**
   * @brief Construct a new Constraint object from the symbolic
   * expression c
   *
   * @param name Optional name for the constraint, if set to "", creates a
   * default name for the constraint base on the id of the constraint
   * @param c The expression to compute the constraint and derivatives from
   * @param bounds Bounds for the constraint (e.g. equality, positive,
   * negative ...)
   */
  Constraint(const std::string &name, const casadi::SX &ex,
             const casadi::SXVector &x, const casadi::SXVector &p,
             const BoundsType &bounds, bool sparse = false) {
    // Create function based on casadi
    f_ = std::make_shared<damotion::casadi::FunctionWrapper>(
        damotion::casadi::CreateFunction({ex}, x, p, sparse));

    // Resize the bounds
    ResizeBounds(ex.size1());
    // Update bounds for the constraint
    SetBounds(bounds);
  }

  /**
   * @brief Construct a new Constraint object using an existing common::Function
   * object.
   *
   * @param name
   * @param f
   * @param bounds
   */
  Constraint(const std::string &name, const common::Function::SharedPtr &f,
             const BoundsType &bounds)
      : f_(std::move(f)) {
    // Resize the bounds
    ResizeBounds(f_->GetOutput(0).toConstVectorXdRef().size());
    // Update bounds for the constraint
    SetBounds(bounds);
  }

  /**
   * @brief Name of the constraint
   *
   * @return const std::string
   */
  const std::string name() const { return name_; }

  /**
   * @brief Dimension of the constraint
   *
   */
  int Dimension() const { return 0; }

  /**
   * @brief Set the name of the constraint
   *
   * @param name
   */
  void SetName(const std::string &name) {
    if (name == "") {
      name_ = "constraint_" + std::to_string(CreateID());
    } else {
      name_ = name;
    }
  }

  void Eval(const common::InputRefVector &x, const common::InputRefVector &p,
            bool check = false) {
    // Evaluate the constraints based on the
    common::InputRefVector in = {};
    for (const auto &xi : x) in.push_back(xi);
    for (const auto &pi : p) in.push_back(pi);
    Eigen::VectorXd one(1.0);
    for (size_t i = 0; i < f_->n_out(); ++i) in.push_back(one);
    // Append flags for evaluating jacobian and hessian
    Eigen::VectorXd d_flag(1.0), h_flag(0.0);
    in.push_back(d_flag);
    in.push_back(h_flag);
    f_->Eval(in, check);
  }

  const GenericEigenMatrix &Vector() { return f_->GetOutput(0); }
  const GenericEigenMatrix &Jacobian() { return f_->GetOutput(1); }
  const GenericEigenMatrix &Hessian() { return f_->GetOutput(2); }

  /**
   * @brief Resizes the bounds the size of the constraint output given by
   * common::Function::ny().
   *
   */
  void ResizeBounds(const size_t &n) {
    // Initialise the bounds of the constraint
    double inf = std::numeric_limits<double>::infinity();
    ub_ = inf * Eigen::VectorXd::Ones(n);
    lb_ = -inf * Eigen::VectorXd::Ones(n);
  }

  /**
   * @brief Set the Bounds type for the constraint.
   *
   * @param type
   */
  void SetBounds(const BoundsType &type) {
    bounds_type_ = type;
    SetBoundsByType(ub_, lb_, bounds_type_);
  }

  /**
   * @brief Sets constraint bounds to a custom interval
   *
   * @param lb
   * @param ub
   */
  void SetBounds(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub) {
    bounds_type_ = BoundsType::kCustom;
    lb_ = lb;
    ub_ = ub;

    // Indicate constraint was updated
    IsUpdated() = true;
  }

  /**
   * @brief The current type of bounds for the constraint
   *
   * @return const BoundsType&
   */
  const BoundsType &GetBoundsType() const { return bounds_type_; }

  /**
   * @brief Constraint lower bound (dim x 1)
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &LowerBound() const { return lb_; }
  Eigen::VectorXd &LowerBound() { return lb_; }

  /**
   * @brief Constraint upper bound (dim x 1)
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &UpperBound() const { return ub_; }
  Eigen::VectorXd &UpperBound() { return ub_; }

  /**
   * @brief Tests whether the p-norm of the constraint is within
   * the threshold eps.
   *
   * @param p The norm of the constraint (use Eigen::Infinity for the
   * infinity norm)
   * @param eps
   * @return true
   * @return false
   */
  bool CheckViolation(const int &p = 2, const double &eps = 1e-6) {
    // Determine if constraint within threshold
    double c_norm = 0.0;
    Eigen::Ref<const Eigen::VectorXd> c = Vector().toConstVectorXdRef();
    if (p == 1) {
      c_norm = c.lpNorm<1>();
    } else if (p == 2) {
      c_norm = c.lpNorm<2>();
    } else if (p == Eigen::Infinity) {
      c_norm = c.lpNorm<Eigen::Infinity>();
    }

    return c_norm <= eps;
  }

  /**
   * @brief Indicates if the constraint has been updated since it was last
   * used, can be set to true and false.
   *
   * @return true
   * @return false
   */
  const bool &IsUpdated() const { return updated_; }
  bool &IsUpdated() { return updated_; }

 protected:
 private:
  // Dimension of the constraint
  int dim_ = 0;

  // Flag to indicate if the constraint has changed since it was used
  bool updated_;

  bool code_generated_ = false;

  // Name of the constraint
  std::string name_;

  BoundsType bounds_type_ = BoundsType::kUnbounded;

  // Constraint lower bound
  Eigen::VectorXd lb_;
  // Constraint upper bound
  Eigen::VectorXd ub_;

  // Function to evaluate the constraint
  common::Function::SharedPtr f_;

  /**
   * @brief Creates a unique id for each constraint
   *
   * @return int
   */
  int CreateID() {
    static int next_id = 0;
    int id = next_id;
    next_id++;
    return id;
  }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_BASE_H */
