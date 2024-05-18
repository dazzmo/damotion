#ifndef COSTS_BASE_H
#define COSTS_BASE_H

#include <casadi/casadi.hpp>

#include "damotion/casadi/codegen.h"
#include "damotion/casadi/eigen.h"
#include "damotion/casadi/function.h"
#include "damotion/optimisation/fwd.h"

namespace damotion {
namespace optimisation {

class CostBase {
 public:
  CostBase() = default;
  ~CostBase() = default;

  CostBase(const std::string &name, const casadi::SX &ex,
           const casadi::SXVector &x, const casadi::SXVector &p,
           bool grd = false, bool hes = false)
      : nx_(x.size()), np_(p.size()) {
    // TODO - Set ID
    // Convert input variables to single vector
    casadi::SX x = casadi::SX::vertcat(x);

    // Create functions to compute the constraint and derivatives given the
    // variables and parameters

    // Input vectors {x, p}
    casadi::SXVector in = x;
    for (const casadi::SX &pi : p) {
      in.push_back(pi);
    }

    // Create functions for each and wrap them
    // Constraint
    SetObjectiveFunction(std::make_shared<damotion::casadi::FunctionWrapper>(
        casadi::Function(name, in, {ex})));
    // Jacobian
    if (grd) {
      // Wrap the functions
      SetGradientFunction(std::make_shared<damotion::casadi::FunctionWrapper>(
          casadi::Function(name + "_grd", in, gradient(ex, x))));
      grd_.resize(nx_);
    }

    // Hessian
    if (hes) {
      // Wrap the functions
      SetHessianFunction(
          std::make_shared<damotion::casadi::FunctionWrapper>(casadi::Function(
              name + "_hes", in, casadi::SX::tril(hessian(ex, x)))));
      hes_.resize(nx_, nx_);
    }
  }

  /**
   * @brief Name of the cost
   *
   * @return const std::string&
   */
  const std::string &name() const { return name_; }

  /**
   * @brief Whether the cost has a non-zero gradient
   *
   * @return true
   * @return false
   */
  bool HasGradient() const { return has_grd_; }

  /**
   * @brief Whether the cost has a non-zero hessian
   *
   * @return true
   * @return false
   */
  bool HasHessian() const { return has_hes_; }

  /**
   * @brief Evaluate the cost with the current input variables and
   * parameters, indicating if gradient is required
   *
   * @param x Variables for the cost
   * @param p Parameters for the cost
   * @param grd Whether to also compute the gradient
   */
  virtual void eval(const common::InputRefVector &x,
                    const common::InputRefVector &p, bool grd = true) const {
    VLOG(10) << this->name() << " eval()";
    common::InputRefVector in = {};
    for (const auto &xi : x) in.push_back(xi);
    for (const auto &pi : p) in.push_back(pi);

    // Call necessary cost functions
    this->fobj_->call(in);
    if (grd) this->fgrd_->call(in);
  }

  /**
   * @brief Evaluate the Hessian of the constraint with respect to the inputs x
   *
   * @param x
   * @param p
   */
  virtual void eval_hessian(const common::InputRefVector &x,
                            const common::InputRefVector &p) const {
    VLOG(10) << this->name() << " eval_hessian()";
    // Create input for the lambda-hessian product
    common::InputRefVector in = {};
    for (const auto &xi : x) in.push_back(xi);
    for (const auto &pi : p) in.push_back(pi);

    // Call necessary constraint functions
    this->fhes_->call(in);
  }

  /**
   * @brief Returns the most recent evaluation of the cost objective
   *
   * @return const double&
   */
  const double &Objective() const { return fobj_->getOutput(0); }
  /**
   * @brief The gradient of the cost with respect to the i-th variable
   * vector
   *
   * @param i
   * @return const VectorXd&
   */
  const GenericMatrixData &Gradient() const {
    assert(has_grd_ && "This cost does not have a gradient to access");
    return fgrd_->getOutput(0);
  }
  /**
   * @brief Returns the Hessian with respect to the variables.
   * Please note that this formulation produces only the lower-triangular.
   *
   * @param i
   * @param j
   * @return const MatrixType&
   */
  const GenericMatrixData &Hessian() const {
    assert(has_hes_ && "This cost does not have a hessian to access");
    return fhes_->getOutput(0);
  }

  /**
   * @brief Number of parameters used to determine the constraint
   *
   * @return const int&
   */
  const int &NumberOfInputParameters() const { return np_; }

 protected:
  /**
   * @brief Set the Objective Function object
   *
   * @param f
   */
  void SetObjectiveFunction(const common::Function::SharedPtr &f) { fobj_ = f; }

  /**
   * @brief Set the Gradient Function object
   *
   * @param f
   */
  void SetGradientFunction(const common::Function::SharedPtr &f) {
    fgrd_ = f;
    has_grd_ = true;
  }

  /**
   * @brief Set the Hessian Function object
   *
   * @param f
   */
  void SetHessianFunction(const typename common::Function::SharedPtr &f) {
    fhes_ = f;
    has_hes_ = true;
  }

  // Flags to indicate if cost can compute derivatives
  bool has_grd_ = false;
  bool has_hes_ = false;

 private:
  // Number of variable inputs
  int nx_ = 0;
  // Number of parameter inputs
  int np_ = 0;

  // Name of the cost
  std::string name_;

  // Objective function
  common::Function::SharedPtr fobj_;
  // Gradient function of the objective
  common::Function::SharedPtr fgrd_;
  // Hessian funciton of the objective
  common::Function::SharedPtr fhes_;

  /**
   * @brief Creates a unique id for each cost
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

#endif /* COSTS_BASE_H */
