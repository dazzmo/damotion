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
    ::casadi::Function fj("df", in, {df});

    // Hessian
    ::casadi::SX hf;
    // Compute lower-triangular hessian matrix
    hf = ::casadi::SX::tril(::casadi::SX::hessian(f, xv));
    // Densify if requested
    if (!sparse) hf = ::casadi::SX::densify(hf);
    ::casadi::Function fh("hf", in, {hf});

    fc_ = std::make_shared<damotion::casadi::FunctionWrapper>(fc);
    fj_ = std::make_shared<damotion::casadi::FunctionWrapper>(fj);
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
  void SetName(const std::string &name) {
    if (name == "") {
      name_ = "cost_" + std::to_string(createID());
    } else {
      name_ = name;
    }
  }

  void eval(const std::vector<ConstVectorRef> &x,
            const std::vector<ConstVectorRef> &p, bool check = false) {
    // Evaluate the constraints based on the
    std::vector<ConstVectorRef> in = {};
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

  const GenericEigenMatrix &Value() { return f_->GetOutput(0); }
  const GenericEigenMatrix &Gradient() { return f_->GetOutput(1); }
  const GenericEigenMatrix &Hessian() { return f_->GetOutput(2); }

 private:
  // Name of the cost
  std::string name_;

  // Function to evaluate the cost
  common::Function::SharedPtr f_;

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
