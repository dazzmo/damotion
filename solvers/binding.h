#ifndef SOLVERS_BINDING_H
#define SOLVERS_BINDING_H

#include <memory>

#include "symbolic/parameter.h"
#include "symbolic/variable.h"

namespace sym = damotion::symbolic;

namespace damotion {
namespace optimisation {

template <typename T>
class Binding {
   public:
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
    }

    Binding(const std::shared_ptr<T> &c,
            const std::vector<sym::VariableVector> &variables,
            const std::vector<const double *> &parameters,
            const std::vector<int> &variable_start_indices,
            const std::vector<std::vector<int>> &variable_jacobian_indices) {
        c_ = c;

        x_ = variables;
        p_ = parameters;

        x_idx_ = variable_start_indices;
        jac_idx_ = variable_jacobian_indices;

        nx_ = variables.size();
        np_ = parameters.size();
    }

    /**
     * @brief Construct a new Binding object of a new type
     *
     * @tparam U
     * @param b
     */
    template <typename U>
    Binding(const Binding<U> &b,
            typename std::enable_if_t<std::is_convertible_v<
                std::shared_ptr<U>, std::shared_ptr<T>>> * = nullptr)
        : Binding(b.GetPtr(), b.GetVariables(), b.GetParameters(),
                  b.VariableStartIndices(), b.SparseJacobianIndices()) {}

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

    void SetVariableStartIndices(const std::vector<int> &indices) {
        assert(indices.size() == x_.size() &&
               "Incorrect number of indices provided\n");
        x_idx_ = indices;
    }

    const std::vector<int> &VariableStartIndices() const { return x_idx_; }
    const std::vector<std::vector<int>> &SparseJacobianIndices() const {
        return jac_idx_;
    }

   private:
    int nx_ = 0;
    int np_ = 0;

    std::shared_ptr<T> c_;
    // Vector of variables bound to the constraint
    std::vector<sym::VariableVector> x_ = {};
    // Vector of pointers to references to parameters bound to the constraint
    std::vector<const double *> p_ = {};

    // Starting indices of the decision variables in x
    std::vector<int> x_idx_;

    // Indices of the jacobian data vectors within the sparse constraint
    // jacobian data vector
    std::vector<std::vector<int>> jac_idx_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_BINDING_H */
