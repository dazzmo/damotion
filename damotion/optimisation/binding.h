#ifndef OPTIMISATION_BINDING_H
#define OPTIMISATION_BINDING_H

#include <memory>

#include "damotion/optimisation/fwd.h"

namespace damotion {
namespace optimisation {

class BindingBase {
 public:
  typedef int Id;

  BindingBase() = default;
  ~BindingBase() = default;

  const Id &id() const { return id_; }

  /**
   * @brief
   *
   * @param c
   * @param x
   * @param p
   */
  BindingBase(const sym::VariableRefVector &x,
              const sym::VariableRefVector &p = {}) {
    // Set variable vector
    x_.reserve(x.size());
    for (auto xi : x) {
      x_.push_back(std::make_shared<sym::VariableVector>(xi));
    }

    // Create vector of Variables
    p_.reserve(p.size());
    for (auto pi : p) {
      p_.push_back(std::make_shared<sym::VariableVector>(pi));
    }

    // Create a concatenated variable vector
    xc_ = std::make_shared<sym::VariableVector>(
        sym::ConcatenateVariableRefVector(x));

    nx_ = x.size();
    np_ = p.size();

    static Id next_id = 0;
    id_ = next_id++;
    VLOG(10) << "Created binding with ID " << id_;
  }

  /**
   * @brief Number of inputs x for the binding
   *
   * @return const int&
   */
  const int &nx() const { return nx_; }

  /**
   * @brief Number of Variables p for the binding
   *
   * @return const int&
   */
  const int &np() const { return np_; }

  /**
   * @brief The i-th input x_i for the binding
   *
   * @param i
   * @return const sym::VariableVector&
   */
  const sym::VariableVector &x(const int i) const {
    assert(i < 0 && i >= nx() && "Out of range for binding inputs");
    return *x_[i];
  }

  /**
   * @brief Returns the vector of concatenated inputs x
   *
   * @return const sym::VariableVector&
   */
  const sym::VariableVector &GetConcatenatedVariableVector() const {
    return *xc_;
  }

  /**
   * @brief The i-th Variable p_i for the binding
   *
   * @param i
   * @return const sym::VariableVector&
   */
  const sym::VariableVector &p(const int &i) const {
    assert(i < 0 && i >= np() && "Out of range for binding Variables");
    return *p_[i];
  }

 protected:
  Id id_;

  int nx_ = 0;
  int np_ = 0;

  // Vector of variables bound to the constraint
  std::vector<std::shared_ptr<sym::VariableVector>> x_ = {};
  std::vector<std::shared_ptr<sym::VariableVector>> p_ = {};
  std::shared_ptr<sym::VariableVector> xc_ = nullptr;
};

template <typename T>
class Binding : public BindingBase {
 public:
  template <typename U>
  friend class Binding;

  Binding() = default;
  ~Binding() = default;

  /**
   * @brief
   *
   * @param c
   * @param x
   * @param p
   */
  Binding(const std::shared_ptr<T> &c, const sym::VariableRefVector &x,
          const sym::VariableRefVector &p = {})
      : BindingBase(x, p) {
    c_ = c;
  }

  /**
   * @brief Cast a binding of type U to a Binding of type T, if convertible.
   *
   * @tparam U
   * @param b
   */
  template <typename U>
  Binding(const Binding<U> &b,
          typename std::enable_if_t<
              std::is_convertible_v<std::shared_ptr<U>, std::shared_ptr<T>>> * =
              nullptr)
      : Binding() {
    // Maintain the same binding id
    id_ = b.id();
    // Copy all data
    c_ = b.c_;

    nx_ = b.nx_;
    x_ = b.x_;
    xc_ = b.xc_;

    np_ = b.np_;
    p_ = b.p_;
  }

  /**
   * @brief Returns the bounded object
   *
   * @return T&
   */
  T &Get() { return *c_; }
  const T &Get() const { return *c_; }

  const std::shared_ptr<T> &GetPtr() const { return c_; }

 private:
  // Pointer to bound class T
  std::shared_ptr<T> c_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* OPTIMISATION_BINDING_H */
