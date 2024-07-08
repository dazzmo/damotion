#ifndef OPTIMISATION_BINDING_H
#define OPTIMISATION_BINDING_H

#include <memory>

#include "damotion/common/logging.h"
#include "damotion/optimisation/fwd.h"

namespace damotion {
namespace optimisation {

class BindingBase {
 public:
  typedef size_t Id;

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
  BindingBase(const sym::VectorRefList &x, const sym::VectorRefList &p = {})
      : x_(x), p_(p) {
    // Create a concatenated variable vector
    xc_ = std::make_shared<sym::Vector>(sym::concatenateVariables(x));

    static Id next_id = 0;
    id_ = next_id++;
    VLOG(10) << "Created binding with ID " << id_;
  }

  /**
   * @brief Variable vectors x for the binding
   *
   * @return sym::VectorRefList
   */
  const sym::VectorRefList &x() const { return x_; }

  /**
   * @brief Variable parameters p for the binding
   *
   * @return sym::VectorRefList
   */
  const sym::VectorRefList &p() const { return p_; }

  /**
   * @brief Returns the vector of concatenated inputs x
   *
   * @return const sym::Vector&
   */
  const sym::Vector &getConcatenatedVector() const { return *xc_; }

 protected:
  Id id_;

  // Vector of variables bound to the constraint
  sym::VectorRefList x_;
  sym::VectorRefList p_;
  std::shared_ptr<sym::Vector> xc_ = nullptr;
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
  Binding(const std::shared_ptr<T> &c, const sym::VectorRefList &x,
          const sym::VectorRefList &p = {})
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
    
    c_ = b.c_;
    
    // Copy all data
    xc_ = b.xc_;
    
    x_ = b.x_;
    p_ = b.p_;
  }

  /**
   * @brief Returns the bounded object
   *
   * @return T&
   */
  T &get() { return *c_; }
  const T &get() const { return *c_; }

  const std::shared_ptr<T> &getPtr() const { return c_; }

 private:
  // Pointer to bound class T
  std::shared_ptr<T> c_;
};

}  // namespace optimisation
}  // namespace damotion

#endif/* OPTIMISATION_BINDING_H */
