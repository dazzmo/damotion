#ifndef CONSTRAINTS_BASE_H
#define CONSTRAINTS_BASE_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>

#include "damotion/casadi/function.hpp"
#include "damotion/core/function.hpp"
#include "damotion/optimisation/bounds.hpp"

namespace damotion {
namespace optimisation {

/**
 * @brief Generic constraint with a single vector input
 *
 */
class Constraint : public FunctionBase<1, Eigen::VectorXd>,
                   public BoundedObject<Eigen::VectorXd> {
 public:
  using Index = std::size_t;

  using SharedPtr = std::shared_ptr<Constraint>;
  using UniquePtr = std::unique_ptr<Constraint>;

  using String = std::string;

  using Base = FunctionBase<1, Eigen::VectorXd>;

  Constraint() : sz_(0), nx_(0), np_(0), name_("") {}

  /**
   * @brief Creates a constraint with name and of size sz. The constraint takes
   * nx variables and np parameters to evaluate it.
   *
   * @param name
   * @param sz
   * @param nx
   * @param np
   */
  Constraint(const String &name, const Index &sz, const Index &nx,
             const Index &np = 0)
      : sz_(sz), nx_(nx), np_(np), name_(name) {
    this->set_parameters(Eigen::VectorXd::Zero(this->np()));
  }

  ~Constraint() = default;

  /**
   * @brief Name of the constraint function
   *
   * @return const String&
   */
  const String &name() const { return name_; }

  /**
   * @brief Size of the constraint vector
   *
   * @return const Index&
   */
  const Index &size() const { return sz_; }

  /**
   * @brief Size of the input vector for the constraint \f$ c(x, p) \f$
   *
   * @return const Index&
   */
  const Index &nx() const { return nx_; }

  /**
   * @brief Size of the parameter vector for the constraint \f$ c(x, p) \f$
   *
   * @return const Index&
   */
  const Index &np() const { return np_; }

  /**
   * @brief Determines whether a constraint is contained within its bounds based
   * on a user-defined threshold.
   *
   * @param val
   * @param con
   * @param norm
   * @return true
   * @return false
   */
  static bool isSatisfied(const Eigen::VectorXd &val, const Constraint &con,
                          const Eigen::Index &norm = 2) {
    constexpr double eps = 1e-4;
    if (con.getBoundsType() == BoundType::EQUALITY) {
      return val.lpNorm<Eigen::Infinity>() < eps;
    } else {
      return (val - con.lb()).lpNorm<Eigen::Infinity>() < eps &&
             (con.ub() - val).lpNorm<Eigen::Infinity>() < eps;
    }
  }

  virtual Eigen::VectorXd evaluate(
      const InputVectorType &x, OptionalJacobianType jac = nullptr) const = 0;

 protected:
  void set_nc(const Index &nc) { sz_ = nc; }
  void set_nx(const Index &nx) { nx_ = nx; }
  void set_np(const Index &np) { np_ = np; }

  void set_name(const String &name) { name_ = name; }

 private:
  Index nx_;
  Index np_;
  Index sz_;

  String name_ = "";
};

// TODO - Create constraint violation function

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_BASE_H */
