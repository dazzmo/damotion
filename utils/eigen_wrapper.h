#ifndef UTILS_EIGEN_WRAPPER_H
#define UTILS_EIGEN_WRAPPER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <casadi/casadi.hpp>

namespace casadi_utils {
namespace eigen {

template <typename T, int rows, int cols>
void toEigen(const casadi::Matrix<T> &C, Eigen::Matrix<T, rows, cols> &E) {}

template <typename T, int rows, int cols>
void toCasadi(const Eigen::Matrix<T, rows, cols> &E, casadi::Matrix<T> &C) {}

template <typename T>
class SparseMatrixWrapper {};

/**
 * @brief Class that takes a casadi::Function and allows inputs and outputs to
 * be extracted as Eigen matrices
 *
 */
class FunctionWrapper {
   public:
    FunctionWrapper() = default;
    FunctionWrapper(casadi::Function f);
    FunctionWrapper &operator=(casadi::Function f);

    FunctionWrapper(const FunctionWrapper & other);
    WrappedFunction& operator=(const WrappedFunction&);

    ~FunctionWrapper() = default;

    /**
     * @brief Sets the i-th input for the function
     *
     * @param i
     * @param x
     */
    void setInput(int i, Eigen::Ref<const Eigen::VectorXd> x);

    /**
     * @brief Calls the function with the current inputs
     *
     * @param sparse
     */
    void call(bool sparse = false);

    /**
     * @brief Indicate that the output will be sparse
     *
     * @param i Index of the output
     */
    void setSparseOutput(int i);

    /**
     * @brief Returns the dense output i
     *
     * @param i
     * @return const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd &getOutput(int i);

    /**
     * @brief Returns the sparse matrix output i. You must call
     * setSparseOutput(i) beforehand to return a sparse output for output i.
     *
     * @param i
     * @return const Eigen::SparseMatrix<double>&
     */
    const Eigen::SparseMatrix<double> &getOutputSparse(int i);

   private:
    // Data input vector for casadi function
    std::vector<const double *> in_data_;
    // Data output from casadi function
    std::vector<std::vector<double>> out_data_;

    // Dense matrix outputs
    std::vector<Eigen::MatrixXd> out_;
    // Sparse matrix outputs
    std::vector<Eigen::SparseMatrix<double>> out_sparse_;

    // Row triplet data for nnz of each output
    std::vector<std::vector<casadi_int>> rows_;
    // Column triplet data for nnz of each output
    std::vector<std::vector<casadi_int>> cols_;

    // Flag to indicate if output has been set as sparse
    std::vector<bool> is_out_sparse_;

    // Underlying function
    casadi::Function f_;

    Eigen::SparseMatrix<double> createSparseMatrix(
        const casadi::Sparsity &sparsity);
};

}  // namespace eigen
}  // namespace casadi_utils

#endif /* UTILS_EIGEN_WRAPPER_H */
