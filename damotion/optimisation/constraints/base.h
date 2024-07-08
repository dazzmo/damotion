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
  Constraint(const std::string &name, const casadi::SX &f,
             const casadi::SXVector &x, const casadi::SXVector &p,
             const Bounds::Type &bounds, bool sparse = false)
      : dim_(f.size1()) {
    // Create concatenated x vector for derivative purposes
    ::casadi::SX xv = ::casadi::SX::vertcat(x);

    // Create input and output vectors
    ::casadi::SXVector in = {}, out = {};

    for (const auto &xi : x) in.push_back(xi);
    for (const auto &pi : p) in.push_back(pi);

    ::casadi::Function fc("f", in, {f});

    // Jacobian
    ::casadi::SX df;
    df = ::casadi::SX::jacobian(f, xv);
    // Densify if requested
    if (!sparse) df = ::casadi::SX::densify(df);
    ::casadi::Function fj("df", in, {df});

    // Hessian
    ::casadi::SX hf;
    ::casadi::SX l = ::casadi::SX::sym("l", dim());
    // Add these multipliers to the input parameters
    in.push_back(l);
    // Compute lower-triangular hessian matrix
    hf = ::casadi::SX::tril(::casadi::SX::hessian(mtimes(l.T(), f), xv));
    // Densify if requested
    if (!sparse) hf = ::casadi::SX::densify(hf);
    ::casadi::Function fh("hf", in, {hf});

    fc_ = std::make_shared<damotion::casadi::FunctionWrapper>(fc);
    fj_ = std::make_shared<damotion::casadi::FunctionWrapper>(fj);
    fh_ = std::make_shared<damotion::casadi::FunctionWrapper>(fh);

    // Resize the bounds
    resizeBounds(dim());
    // Update bounds for the constraint
    setBounds(bounds);
  }

  /**
   * @brief Construct a new Constraint object using an existing common::Function
   * object with the ability to compute the constraint.
   *
   * @param name
   * @param f
   * @param bounds
   */
  Constraint(const std::string &name, const common::Function::SharedPtr &f,
             const Bounds::Type &bounds,
             const common::Function::SharedPtr &fjac = nullptr,
             const common::Function::SharedPtr &fhes = nullptr)
      : dim_(f->getOutput(0).asDense().rows()),
        fc_({std::move(f)}),
        fj_({std::move(fjac)}),
        fh_({std::move(fhes)}) {
    // Resize the bounds
    resizeBounds(dim());
    // Update bounds for the constraint
    setBounds(bounds);
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
  size_t dim() const { return dim_; }

  /**
   * @brief Whether the constraint provides the ability to compute its Jacobian
   *
   * @return true
   * @return false
   */
  bool hasJacobian() const { return fj_ != nullptr; }

  /**
   * @brief Whether the constraint provides the ability to compute its Hessian
   * (based on the multipler-constraint product)
   *
   * @return true
   * @return false
   */
  bool hasHessian() const { return fh_ != nullptr; }

  /**
   * @brief Set the name of the constraint
   *
   * @param name
   */
  void setName(const std::string &name) {
    if (name == "") {
      name_ = "constraint_" + std::to_string(createID());
    } else {
      name_ = name;
    }
  }

  void eval(const common::Function::InputVector &x,
            const common::Function::InputVector &p, bool jac) {
    // Evaluate the constraints based on the
    common::Function::InputVector in = {};
    for (const auto &xi : x) in.push_back(xi);
    for (const auto &pi : p) in.push_back(pi);
    // Perform evaluation depending on what method is used
    fc_->eval(in);
    if (jac) fj_->eval(in);
  }

  void eval_h(const common::Function::InputVector &x,
              const common::Function::InputVector &p, const ConstVectorRef &l) {
    // Evaluate the constraints based on the
    common::Function::InputVector in = {};
    for (const auto &xi : x) in.push_back(xi);
    for (const auto &pi : p) in.push_back(pi);
    in.push_back(l);
    fh_->eval(in);
  }

  /**
   * @brief Constraint vector
   *
   * @return const common::Function::Output&
   */
  const common::Function::Output &con() const { return fc_->getOutput(0); }

  /**
   * @brief Constraint Jacobian
   *
   * @return const common::Function::Output&
   */
  const common::Function::Output &jac() const { return fj_->getOutput(0); }

  /**
   * @brief Constraint-multiplier Hessian
   *
   * @return const common::Function::Output&
   */
  const common::Function::Output &hes() const { return fh_->getOutput(0); }

  /**
   * @brief Resizes the bounds the size of the constraint output given by
   * common::Function::ny().
   *
   */
  void resizeBounds(const size_t &n) {
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
  void setBounds(const Bounds::Type &type) {
    bounds_type_ = type;
    setBoundsByType(ub_, lb_, bounds_type_);
  }

  /**
   * @brief Sets constraint bounds to a custom interval
   *
   * @param lb
   * @param ub
   */
  void setBounds(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub) {
    bounds_type_ = Bounds::Type::kCustom;
    lb_ = lb;
    ub_ = ub;

    // Indicate constraint was updated
    isUpdated() = true;
  }

  /**
   * @brief The current type of bounds for the constraint
   *
   * @return const Bounds::Type&
   */
  const Bounds::Type &getBoundsType() const { return bounds_type_; }

  /**
   * @brief Constraint lower bound (dim x 1)
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &lowerBound() const { return lb_; }
  Eigen::VectorXd &lowerBound() { return lb_; }

  /**
   * @brief Constraint upper bound (dim x 1)
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &upperBound() const { return ub_; }
  Eigen::VectorXd &upperBound() { return ub_; }

  /**
   * @brief Indicates if the constraint has been updated since it was last
   * used, can be set to true and false.
   *
   * @return true
   * @return false
   */
  const bool &isUpdated() const { return updated_; }
  bool &isUpdated() { return updated_; }

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
  bool checkConstraintViolation(const int &p = 2, const double &eps = 1e-6) {
    // Determine if constraint within threshold
    double c_norm = 0.0;
    if (p == 1) {
      c_norm = fc_->getOutput(0).asDense().lpNorm<1>();
    } else if (p == 2) {
      c_norm = fc_->getOutput(0).asDense().lpNorm<2>();
    } else if (p == Eigen::Infinity) {
      c_norm = fc_->getOutput(0).asDense().lpNorm<Eigen::Infinity>();
    }

    return c_norm <= eps;
  }

 protected:
 private:
  // dimension of the constraint
  size_t dim_ = 0;

  // Flag to indicate if the constraint has changed since it was used
  bool updated_;

  bool code_generated_ = false;

  // Name of the constraint
  std::string name_;

  Bounds::Type bounds_type_ = Bounds::Type::kUnbounded;

  // Constraint lower bound
  Eigen::VectorXd lb_;
  // Constraint upper bound
  Eigen::VectorXd ub_;

  // Vector of function pointers for evaluation
  common::Function::SharedPtr fc_;
  // Function pointer for constraint Jacobian evaluation
  common::Function::SharedPtr fj_;
  // Function pointer for multiplier-constraint product Hessian evaluation
  common::Function::SharedPtr fh_;

  /**
   * @brief Creates a unique id for each constraint
   *
   * @return int
   */
  int createID() {
    static int next_id = 0;
    int id = next_id;
    next_id++;
    return id;
  }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_BASE_H */
