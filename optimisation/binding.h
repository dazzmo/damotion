#ifndef OPTIMISATION_BINDING_H
#define OPTIMISATION_BINDING_H

#include <memory>

#include "symbolic/parameter.h"
#include "symbolic/variable.h"

namespace sym = damotion::symbolic;

namespace damotion {
namespace optimisation {

template <typename T>
class Binding {
   public:
    typedef int Id;
    typedef std::shared_ptr<sym::Parameter> ParameterPtr;
    typedef std::vector<ParameterPtr> ParameterPtrVector;

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
    Binding(const std::shared_ptr<T> &c, const sym::VariableVectorRef &x,
            const sym::ParameterVector &p = {}) {
        c_ = c;
        // Set variable vector
        x_ = std::make_shared<sym::VariableVector>(x);
        // Create vector of parameters
        p_.reserve(p.size());
        for (auto pi : p) {
            p_.push_back(std::make_shared<sym::Parameter>(pi));
        }

        np_ = p.size();

        SetId();
    }

    Binding(const std::shared_ptr<T> &c, const sym::VariableVector::SharedPtr &x,
            const ParameterPtrVector &p)
        : x_(x), p_(p) {
        // Copy binding pointer
        c_ = c;
        // Set sizes for the bound variables and parameters
        np_ = parameters.size();
        // Set an ID
        SetId();
    }

    /**
     * @brief Cast a binding of type U to a Binding of type T, if convertible.
     *
     * @tparam U
     * @param b
     */
    template <typename U>
    Binding(const Binding<U> &b,
            typename std::enable_if_t<std::is_convertible_v<
                std::shared_ptr<U>, std::shared_ptr<T>>> * = nullptr)
        : Binding(b.GetPtr(), b.GetVariables(), b.GetParameters()) {
        // Maintain the same binding id
        id_ = b.id();
    }

    const int &NumberOfParameters() const { return np_; }

    /**
     * @brief Returns the bounded object
     *
     * @return T&
     */
    T &Get() { return *c_; }
    const T &Get() const { return *c_; }

    const std::shared_ptr<T> &GetPtr() const { return c_; }

    /**
     * @brief Get the variable vector associated with the binding
     *
     * @return const sym::VariableVector&
     */
    const sym::VariableVector &GetVariableVector() const { return *x_; }

    const ParameterPtrVector &GetParameters() const { return p_; }
    const sym::Parameter &GetParameter(const int &i) const { return *p_[i]; }

   private:
    Id id_;

    int np_ = 0;

    std::shared_ptr<T> c_;

    // Vector of variables bound to the constraint
    std::shared_ptr<sym::VariableVector> x_ = nullptr;
    ParameterPtrVector p_ = {};

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

#endif /* OPTIMISATION_BINDING_H */
