#ifndef COSTS_BASE_H
#define COSTS_BASE_H

#include <casadi/casadi.hpp>

#include "damotion/casadi/function.hpp"

namespace damotion {
namespace optimisation {

/**
 * @brief Generic cost with a single vector input
 *
 */
class Cost : public FunctionBase<1, double> {
 public:
  using SharedPtr = std::shared_ptr<Cost>;
  using UniquePtr = std::unique_ptr<Cost>;

  using Id = Index;

  using String = std::string;

  using Base = FunctionBase<1, double>;

  const String &name() const { return name_; }

  Cost(const String &name, const Index &nx, const Index &np = 0)
      : Base(), nx_(nx), np_(np) {}

  /**
   * @brief Size of the input vector for the cost \f$ c(x, p) \f$
   *
   * @return const Index&
   */
  const Index &nx() const { return nx_; }

  /**
   * @brief Size of the parameter vector for the cost \f$ c(x, p) \f$
   *
   * @return const Index&
   */
  const Index &np() const { return np_; }

  virtual double evaluate(const InputVectorType &x,
                          OptionalJacobianType grd = nullptr) const = 0;

 protected:
  void set_nx(const Index &nx) { nx_ = nx; }
  void set_np(const Index &np) { np_ = np; }

 private:
  Index nx_;
  Index np_;

  String name_ = "";

  /**
   * @brief Creates a unique id for each cost
   *
   * @return Id
   */
  Id createID() {
    static Id next_id = 0;
    Id id = next_id;
    next_id++;
    return id;
  }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* COSTS_BASE_H */
