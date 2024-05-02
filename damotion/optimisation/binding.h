#ifndef OPTIMISATION_BINDING_H
#define OPTIMISATION_BINDING_H

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
    typedef std::shared_ptr<sym::Parameter> ParameterPtr;
    typedef std::vector<ParameterPtr> ParameterPtrVector;

    template<typename U>
    friend class Binding;

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
            const sym::ParameterVector &p = {}) {
        c_ = c;
        // Set variable vector
        x_.reserve(x.size());
        for (auto xi : x) {
            x_.push_back(std::make_shared<sym::VariableVector>(xi));
        }

        // Create vector of parameters
        p_.reserve(p.size());
        for (auto pi : p) {
            p_.push_back(std::make_shared<sym::Parameter>(pi));
        }

        // Create a concatenated variable vector
        xc_ = std::make_shared<sym::VariableVector>(
            sym::ConcatenateVariableRefVector(x));

        nx_ = x.size();
        np_ = p.size();

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

    const int &nx() const { return nx_; }
    const int &np() const { return np_; }

    /**
     * @brief Returns the bounded object
     *
     * @return T&
     */
    T &Get() { return *c_; }
    const T &Get() const { return *c_; }

    const std::shared_ptr<T> &GetPtr() const { return c_; }

    const sym::VariableVector &x(const int i) const { return *x_[i]; }
    const sym::VariableVector &GetConcatenatedVariableVector() const {
        return *xc_;
    }

    const sym::Parameter &p(const int &i) const { return *p_[i]; }

   private:
    Id id_;

    int nx_ = 0;
    int np_ = 0;

    std::shared_ptr<T> c_;

    // Vector of variables bound to the constraint
    std::vector<std::shared_ptr<sym::VariableVector>> x_ = {};
    std::shared_ptr<sym::VariableVector> xc_ = nullptr;
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
