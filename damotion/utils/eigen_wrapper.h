#ifndef UTILS_EIGEN_WRAPPER_H
#define UTILS_EIGEN_WRAPPER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <casadi/casadi.hpp>
#include <pinocchio/autodiff/casadi.hpp>

#include "damotion/common/function.h"
#include "damotion/common/sparsity.h"

namespace damotion {
namespace utils {
namespace casadi {

/**
 * @brief Convert a casadi::Matrix<T> matric to an Eigen::Matrix<T>
 *
 * @tparam T
 * @tparam rows
 * @tparam cols
 * @param C
 * @param E
 */
template <typename T, int rows, int cols>
void toEigen(const ::casadi::Matrix<T> &C, Eigen::Matrix<T, rows, cols> &E) {
    E.setZero(C.rows(), C.columns());
    for (int i = 0; i < C.rows(); ++i) {
        for (int j = 0; j < C.columns(); ++j) {
            E(i, j) = T(C(i, j));
        }
    }
}

/**
 * @brief Convert an Eigen::Matrix<T> object to a casadi::Matrix<T> object
 *
 * @tparam T
 * @tparam rows
 * @tparam cols
 * @param E
 * @param C
 */
template <typename T, int rows, int cols>
void toCasadi(const Eigen::Matrix<T, rows, cols> &E, ::casadi::Matrix<T> &C) {
    C.resize(E.rows(), E.cols());
    for (int i = 0; i < E.rows(); ++i) {
        for (int j = 0; j < E.cols(); ++j) {
            // Only fill in non-zero entries
            if (!::casadi::is_zero(E(i, j))) {
                C(i, j) = E(i, j);
            }
        }
    }
}

/**
 * @brief Convert a casadi::Matrix<T> object to an
 * Eigen::Matrix<casadi::Matrix<T>> object (e.g. convert a casadi::SX to an
 * Eigen::Matrix<casadi::SX>)
 *
 * @tparam T
 * @tparam rows
 * @tparam cols
 * @param C
 * @param E
 */
template <typename T, int rows, int cols>
void toEigen(const ::casadi::Matrix<T> &C,
             Eigen::Matrix<::casadi::Matrix<T>, rows, cols> &E) {
    E.setZero(C.rows(), C.columns());
    for (int i = 0; i < C.rows(); ++i) {
        for (int j = 0; j < C.columns(); ++j) {
            E(i, j) = ::casadi::Matrix<T>(C(i, j));
        }
    }
}

template <typename T, int rows, int cols>
void toCasadi(const Eigen::Matrix<::casadi::Matrix<T>, rows, cols> &E,
              ::casadi::Matrix<T> &C) {
    C.resize(E.rows(), E.cols());
    for (int i = 0; i < E.rows(); ++i) {
        for (int j = 0; j < E.cols(); ++j) {
            // Only fill in non-zero entries
            if (!::casadi::is_zero(E(i, j)->at(0))) {
                C(i, j) = E(i, j)->at(0);
            }
        }
    }
}

/**
 * @brief Function wrapper base class for casadi functions to Eigen
 * representation
 *
 */
template <typename MatrixType>
class FunctionWrapper : public common::Function<MatrixType> {
   public:
    using SharedPtr = std::shared_ptr<common::Function<MatrixType>>;

    FunctionWrapper() = default;
    ~FunctionWrapper() {
        // Release memory for casadi function
        if (!f_.is_null()) {
            f_.release(mem_);
        }
    }

    FunctionWrapper(const ::casadi::Function &f)
        : common::Function<MatrixType>(f.n_in(), f.n_out()) {
        *this = f;
    }

    FunctionWrapper(const FunctionWrapper &other) { *this = other.f_; }

    FunctionWrapper &operator=(::casadi::Function f) {
        if (f.is_null()) {
            return *this;
        }

        // Copy function
        this->f_ = f;

        this->SetNumberOfInputs(f.n_in());
        this->SetNumberOfOutputs(f.n_out());

        // Initialise output data
        this->OutputVector() = {};
        this->out_data_ptr_ = {};

        // Checkout memory object for function
        this->mem_ = f.checkout();

        // Resize work vectors
        this->in_data_ptr_.assign(f_.sz_arg(), nullptr);

        this->iw_.assign(f_.sz_iw(), 0);
        this->dw_.assign(f_.sz_w(), 0.0);

        // Create dense matrices for the output
        for (int i = 0; i < f_.n_out(); ++i) {
            const ::casadi::Sparsity &sparsity = f_.sparsity_out(i);
            // Create dense matrix for output and add data to output data
            // pointer vector
            this->OutputVector().push_back(
                Eigen::MatrixXd::Zero(sparsity.rows(), sparsity.columns()));
            this->out_data_ptr_.push_back(this->OutputVector().back().data());

            VLOG(10) << f.name() << " Dense Output " << i;
            VLOG(10) << this->OutputVector().back();
        }

        return *this;
    }

    FunctionWrapper &operator=(const FunctionWrapper &other) {
        *this = other.f_;
        return *this;
    }

    /**
     * @brief The casadi::Function that is wrapped.
     *
     * @return casadi::Function&
     */
    ::casadi::Function &f() { return f_; }

    /**
     * @brief Calls the function with the current inputs
     *
     */
    void callImpl(const common::InputRefVector &input) override {
        // Set vector of inputs
        for (int i = 0; i < input.size(); ++i) {
            in_data_ptr_[i] = input[i].data();
        }
        // Call the function
        f_(in_data_ptr_.data(), out_data_ptr_.data(), iw_.data(), dw_.data(),
           mem_);
    }

   protected:
    // Data input vector for casadi function
    mutable std::vector<const double *> in_data_ptr_;
    // Data output pointers for casadi function
    mutable std::vector<double *> out_data_ptr_;

    // Row triplet data for nnz of each output
    std::vector<std::vector<casadi_int>> rows_;
    // Column triplet data for nnz of each output
    std::vector<std::vector<casadi_int>> cols_;

    // Memory allocated for function evaluation
    int mem_;

    // Integer working vector
    mutable std::vector<casadi_int> iw_;
    // Double working vector
    mutable std::vector<double> dw_;

    // Underlying function
    mutable ::casadi::Function f_;
};

// Class specialisations
template <>
FunctionWrapper<double> &FunctionWrapper<double>::operator=(
    ::casadi::Function f);

template <>
FunctionWrapper<Eigen::SparseMatrix<double>> &
FunctionWrapper<Eigen::SparseMatrix<double>>::operator=(::casadi::Function f);

}  // namespace casadi
}  // namespace utils
}  // namespace damotion

#endif /* UTILS_EIGEN_WRAPPER_H */
