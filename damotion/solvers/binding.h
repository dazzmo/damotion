#ifndef SOLVERS_BINDING_H
#define SOLVERS_BINDING_H

#include <memory>

#include "damotion/symbolic/parameter.h"
#include "damotion/symbolic/variable.h"

namespace sym = damotion::symbolic;

namespace damotion {
namespace optimisation {

template <typename T>
class Binding {
 public:
  typedef int Id;

  Binding() = default;
  ~Binding() = default;

  const Id &id() const { return id_; }

  /**
   * @brief
   *
   * @param c
   * @param x
   * @param p
   */
  Binding(const std::shared_ptr<T> &c, const sym::VariableRefVector &x,
          const sym::ParameterRefVector &p = {}) {
    c_ = c;
    // Create variable and parameter vectors
    x_.reserve(x.size());
    p_.reserve(p.size());
    for (auto xi : x) {
      x_.push_back(xi);
    }
    for (auto pi : p) {
      p_.push_back(pi.data());
    }

    nx_ = x.size();
    np_ = p.size();

    SetId();
  }

  Binding(const std::shared_ptr<T> &c,
          const std::vector<sym::VariableVector> &variables,
          const std::vector<const double *> &parameters) {
    c_ = c;

    x_ = variables;
    p_ = parameters;

    nx_ = variables.size();
    np_ = parameters.size();

    SetId();
  }

  /**
   * @brief Construct a new Binding object of a new type.
   *
   * @tparam U
   * @param b
   */
  template <typename U>
  Binding(const Binding<U> &b,
          typename std::enable_if_t<
              std::is_convertible_v<std::shared_ptr<U>, std::shared_ptr<T>>> * =
              nullptr)
      : Binding(b.GetPtr(), b.GetVariables(), b.GetParameters()) {
    // Keep ID the same
    id_ = b.id();
  }

  const int &NumberOfVariables() const { return nx_; }
  const int &NumberOfParameters() const { return np_; }

  /**
   * @brief Returns the bounded object
   *
   * @return T&
   */
  T &Get() { return *c_; }
  const std::shared_ptr<T> &GetPtr() const { return c_; }

  const std::vector<sym::VariableVector> GetVariables() const { return x_; }
  const std::vector<const double *> GetParameters() const { return p_; }

  const sym::VariableVector &GetVariable(const int &i) const { return x_[i]; }
  const double *GetParameterPointer(const int &i) const { return p_[i]; }

 private:
  Id id_;

  int nx_ = 0;
  int np_ = 0;

  std::shared_ptr<T> c_;
  // Vector of variables bound to the constraint
  std::vector<sym::VariableVector> x_ = {};
  // Vector of pointers to references to parameters bound to the constraint
  std::vector<const double *> p_ = {};

  /**
   * @brief Set an ID for the binding, useful for distinguishing one binding
   * from another.
   *
   * @param id
   */
  void SetId() {
    static Id next_id = 0;
    id_ = next_id++;
  }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_BINDING_H */
