#ifndef COSTS_BASE_H
#define COSTS_BASE_H

#include <casadi/casadi.hpp>

#include "damotion/casadi/codegen.h"
#include "damotion/casadi/eigen.h"
#include "damotion/casadi/function.h"
#include "damotion/optimisation/fwd.h"

namespace damotion {
namespace optimisation {

class Cost {
 public:
  Cost() = default;
  ~Cost() = default;

  Cost(const int &nx, const int &np = 0) {}

  /**
   * @brief Construct a new Cost object using an existing common::Function
   * object with the ability to compute the objective.
   *
   * @param name
   * @param f
   * @param bounds
   */
  Cost(const std::string &name, const common::Function::SharedPtr &f,
       const common::Function::SharedPtr &fgrd = nullptr,
       const common::Function::SharedPtr &fhes = nullptr)
      : fc_({std::move(f)}), fg_({std::move(fgrd)}), fh_({std::move(fhes)}) {}

  Cost(const casadi::SX &f, const casadi::SXVector &x,
       const casadi::SXVector &p, bool sparse = false) {
    // Create function based on casadi
    // Create concatenated x vector for derivative purposes
    ::casadi::SX xv = ::casadi::SX::vertcat(x);

    // Create input and output vectors
    ::casadi::SXVector in = {}, out = {};

    for (const auto &xi : x) in.push_back(xi);
    for (const auto &pi : p) in.push_back(pi);

    ::casadi::Function fc("f", in, {f});

    // Jacobian
    ::casadi::SX df;
    df = ::casadi::SX::gradient(f, xv);
    // Densify if requested
    if (!sparse) df = ::casadi::SX::densify(df);
    ::casadi::Function fg("df", in, {df});

    // Hessian
    ::casadi::SX hf;
    // Compute lower-triangular hessian matrix
    hf = ::casadi::SX::tril(::casadi::SX::hessian(f, xv));
    // Densify if requested
    if (!sparse) hf = ::casadi::SX::densify(hf);
    ::casadi::Function fh("hf", in, {hf});

    fc_ = std::make_shared<damotion::casadi::FunctionWrapper>(fc);
    fj_ = std::make_shared<damotion::casadi::FunctionWrapper>(fg);
    fh_ = std::make_shared<damotion::casadi::FunctionWrapper>(fh);
  }

  /**
   * @brief Name of the cost
   *
   * @return const std::string&
   */
  const std::string &name() const { return name_; }

  /**
   * @brief Set the name of the constraint
   *
   * @param name
   */
  void setName(const std::string &name) {
    if (name == "") {
      name_ = "cost_" + std::to_string(createID());
    } else {
      name_ = name;
    }
  }

  void eval(const std::vector<ConstVectorRef> &x,
            const std::vector<ConstVectorRef> &p, bool grd) {
    // Evaluate the constraints based on the
    InputDataVector in = {};
    for (const auto &xi : x) in.push_back(xi.data());
    for (const auto &pi : p) in.push_back(pi.data());
    // Perform evaluation depending on what method is used
    fc_->eval(in);
    if (jac) fj_->eval(in);
  }

  bool hasGradient() const { return fg_ != nullptr; }
  bool hasHessian() const { return fh_ != nullptr; }

  /**
   * @brief Objective
   *
   * @return const Function::Output&
   */
  const Function::Output &obj() const { return fc_->GetOutput(0); }

  /**
   * @brief Objective gradient
   *
   * @return const Function::Output&
   */
  const Function::Output &grd() const { return fg_->GetOutput(0); }

  /**
   * @brief Objective Hessian
   *
   * @return const Function::Output&
   */
  const Function::Output &hes() const { return fh_->GetOutput(0); }

 private:
  // Name of the cost
  std::string name_;

  // Objective function
  common::Function::SharedPtr fc_;
  // Objective gradient
  common::Function::SharedPtr fg_;
  // Objective hessian
  common::Function::SharedPtr fh_;

  /**
   * @brief Creates a unique id for each cost
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

#endif /* COSTS_BASE_H */
