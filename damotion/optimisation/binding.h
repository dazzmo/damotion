#ifndef OPTIMISATION_BINDING_H
#define OPTIMISATION_BINDING_H

#include <memory>

#include "damotion/core/logging.hpp"
#include "damotion/optimisation/fwd.h"

namespace damotion {
namespace optimisation {

class BindingBase {
 public:
  typedef std::size_t Id;

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
  BindingBase(const sym::Vector &x, const sym::Vector &p = {}) : x_(x), p_(p) {
    static Id next_id = 0;
    id_ = next_id++;
    VLOG(10) << "Created binding with ID " << id_;
  }

  /**
   * @brief Variable vectors x for the binding
   *
   * @return sym::Vector
   */
  const sym::Vector &x() const { return x_; }

  /**
   * @brief Variable parameters p for the binding
   *
   * @return sym::Vector
   */
  const sym::Vector &p() const { return p_; }

 protected:
  Id id_;

  // Vector of variables bound to the constraint
  sym::Vector x_;
  sym::Vector p_;
};

template <class ObjectType>
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
  Binding(const std::shared_ptr<ObjectType> &f, const sym::Vector &x,
          const sym::Vector &p = {})
      : BindingBase(x, p), obj_(std::move(f)) {}

  /**
   * @brief Cast a binding of type U to a Binding of type T, if convertible.
   *
   * @tparam U
   * @param b
   */
  template <typename U>
  Binding(const Binding<U> &b,
          typename std::enable_if_t<std::is_convertible_v<
              std::shared_ptr<U>, std::shared_ptr<ObjectType>>> * = nullptr)
      : Binding() {
    // Maintain the same binding id
    id_ = b.id();

    // Convert to new pointer type
    obj_ = static_cast<std::shared_ptr<ObjectType>>(b.obj_);

    // Copy vectors
    x_ = b.x_;
    p_ = b.p_;
  }

  /**
   * @brief Returns the bounded object
   *
   * @return T&
   */
  ObjectType &get() { return *obj_; }
  const ObjectType &get() const { return *obj_; }

  const std::shared_ptr<ObjectType> &getPtr() const { return obj_; }

 private:
  std::shared_ptr<ObjectType> obj_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* OPTIMISATION_BINDING_H */
