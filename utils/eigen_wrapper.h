#ifndef UTILS_EIGEN_WRAPPER_H
#define UTILS_EIGEN_WRAPPER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <casadi/casadi.hpp>
#include <pinocchio/autodiff/casadi.hpp>

#include "common/function.h"

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

template <typename T>
class SparseMatrixWrapper {};

/**
 * @brief Class that takes a casadi::Function and allows inputs and outputs to
 * be extracted as Eigen matrices
 *
 */
class FunctionWrapper : public common::Function {
   public:
    FunctionWrapper() = default;
    FunctionWrapper(::casadi::Function f);
    FunctionWrapper &operator=(::casadi::Function f);

    FunctionWrapper(const FunctionWrapper &other);
    FunctionWrapper &operator=(const FunctionWrapper &other);

    ~FunctionWrapper();

    /**
     * @brief Calls the function with the current inputs
     *
     */
    void callImpl(const common::Function::InputRefVector &input) override;

    void setSparseOutput(int i);

    /**
     * @brief The casadi::Function that is wrapped.
     *
     * @return casadi::Function&
     */
    ::casadi::Function &f() { return f_; }

   private:
    // Data input vector for casadi function
    std::vector<const double *> in_data_ptr_;
    // Data output pointers for casadi function
    std::vector<double *> out_data_ptr_;

    // Row triplet data for nnz of each output
    std::vector<std::vector<casadi_int>> rows_;
    // Column triplet data for nnz of each output
    std::vector<std::vector<casadi_int>> cols_;

    // Memory allocated for function evaluation
    int mem_;

    // Integer working vector
    std::vector<casadi_int> iw_;
    // Double working vector
    std::vector<double> dw_;

    // Underlying function
    ::casadi::Function f_;

    Eigen::SparseMatrix<double> createSparseMatrix(
        const ::casadi::Sparsity &sparsity, std::vector<casadi_int> &rows,
        std::vector<casadi_int> &cols);
};

}  // namespace casadi
}  // namespace utils
}  // namespace damotion

#endif /* UTILS_EIGEN_WRAPPER_H */
