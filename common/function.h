#ifndef COMMON_FUNCTION_H
#define COMMON_FUNCTION_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <functional>
#include <iostream>

#include "common/logging.h"

namespace damotion {
namespace common {

class FunctionBase {
   public:
    /**
     * @brief Vector of input vector references to the function
     *
     */
    typedef std::vector<Eigen::Ref<const Eigen::VectorXd>> InputRefVector;

    FunctionBase() : n_in_(0), n_out_(0) {}

    FunctionBase(const int n_in, const int n_out)
        : n_in_(n_in), n_out_(n_out) {}

    ~FunctionBase() = default;

    /**
     * @brief Number of inputs for the function
     *
     * @return const int
     */
    const int n_in() const { return n_in_; }

    /**
     * @brief Number of outputs for the function
     *
     * @return const int
     */
    const int n_out() const { return n_out_; }

   protected:
    void SetNumberOfInputs(const int &n) { n_in_ = n; }
    void SetNumberOfOutputs(const int &n) { n_out_ = n; }

    /**
     * @brief Assesses if all inputs provided to the function are valid, such as
     * not including infinite values, NaNs...
     *
     * @param input
     * @return true
     * @return false
     */
    bool CheckInputRefVector(const InputRefVector &input) {
        int idx = 0;
        for (const Eigen::Ref<const Eigen::VectorXd> &x : input) {
            if (x.hasNaN() || !x.allFinite()) {
                std::ostringstream ss;
                ss << "Input " << idx << " has invalid values:\n"
                   << x.transpose().format(3);
                throw std::runtime_error(ss.str());
            }
            idx++;
        }

        return true;
    }

   private:
    int n_in_;
    int n_out_;
};

/**
 * @brief Function class that computes dense outputs.
 *
 */
class Function : public FunctionBase {
   public:
    Function() = default;

    Function(const int n_in, const int n_out) : FunctionBase(n_in, n_out) {}

    ~Function() = default;

    /**
     * @brief Pure virtual method that updates the function based on its inputs
     *
     * @param input List of input vectors for the function
     * @param check Perform checks on the input to ensure correct size and good
     * data
     */
    void call(const FunctionBase::InputRefVector &input, bool check = false) {
        if (check) CheckInputRefVector(input);
        callImpl(input);
    }

    template <typename T = Eigen::MatrixXd>
    const Eigen::Ref<const T> getOutput(int i) {
        return out_[i];
    }

   protected:
    /**
     * @brief Virtual call method for derived class to override
     *
     * @param input
     */
    virtual void callImpl(const InputRefVector &input) = 0;

    /**
     * @brief Vector of output matrices
     *
     * @return std::vector<Eigen::MatrixXd>&
     */
    std::vector<Eigen::MatrixXd> &OutputVector() { return out_; }

   private:
    std::vector<Eigen::MatrixXd> out_;
};

/**
 * @brief Function class that computes sparse outputs.
 *
 */
class SparseFunction : public FunctionBase {
   public:
    /**
     * @brief Vector of input vector references to the function
     *
     */
    typedef std::vector<Eigen::Ref<const Eigen::VectorXd>> InputRefVector;

    SparseFunction() : FunctionBase() {}

    SparseFunction(const int n_in, const int n_out)
        : FunctionBase(n_in, n_out) {
        out_.resize(n_out);
    }

    ~SparseFunction() = default;

    /**
     * @brief Pure virtual method that updates the function based on its inputs
     *
     * @param input List of input vectors for the function
     * @param check Perform checks on the input to ensure correct size and good
     * data
     */
    void call(const InputRefVector &input, bool check = false) {
        if (check) CheckInputRefVector(input);
        callImpl(input);
    }

    template <typename T = Eigen::SparseMatrix<double>>
    const Eigen::Ref<const T> getOutput(int i) {
        return out_[i];
    }

   protected:
    /**
     * @brief Virtual call method for derived class to override
     *
     * @param input
     */
    virtual void callImpl(const InputRefVector &input) = 0;

    /**
     * @brief Vector of sparse output matrices
     *
     * @return std::vector<Eigen::MatrixXd>&
     */
    std::vector<Eigen::SparseMatrix<double>> &OutputVector() { return out_; }

   private:
    // Sparse matrix outputs
    std::vector<Eigen::SparseMatrix<double>> out_;
};

/**
 * @brief Function that operates by callback
 *
 */
class CallbackFunction : public Function {
   public:
    typedef std::function<void(const FunctionBase::InputRefVector &,
                               std::vector<Eigen::MatrixXd> &)>
        f_callback_;

    CallbackFunction() = default;
    ~CallbackFunction() = default;

    CallbackFunction(const int n_in, const int n_out,
                     const f_callback_ &callback)
        : Function(n_in, n_out) {
        SetCallback(callback);
    }

    /**
     * @brief Set the size of the output matrix to a dense matrix of size (rows
     * x cols).
     *
     * @param i
     * @param rows
     * @param cols
     */
    void setOutputSize(int i, const int rows, const int cols) {
        assert(i < n_out() && "Number of outputs exceeded");
        OutputVector()[i] = Eigen::MatrixXd::Zero(rows, cols);
    }

    /**
     * @brief Set the callback to be used when call() is used for the function.
     *
     * @param callback
     */
    void SetCallback(const f_callback_ &callback) { f_ = callback; }

    /**
     * @brief Calls the callback function provided to it
     *
     * @param input
     */
    void callImpl(const FunctionBase::InputRefVector &input) override {
        if (f_ == nullptr) {
            throw std::runtime_error(
                "Function callback not provided to CallbackFunction!");
        }
        f_(input, OutputVector());
    }

   private:
    f_callback_ f_ = nullptr;
};

/**
 * @brief Function that operates by callback
 *
 */
class SparseCallbackFunction : public SparseFunction {
   public:
    typedef std::function<void(const FunctionBase::InputRefVector &,
                               std::vector<Eigen::SparseMatrix<double>> &)>
        f_callback_;

    SparseCallbackFunction() = default;
    ~SparseCallbackFunction() = default;

    SparseCallbackFunction(const int n_in, const int n_out,
                           const f_callback_ &callback)
        : SparseFunction(n_in, n_out) {
        SetCallback(callback);
    }

    /**
     * @brief Set the callback to be used when call() is used for the function.
     *
     * @param callback
     */
    void SetCallback(const f_callback_ &callback) { f_ = callback; }

    /**
     * @brief Calls the callback function provided to it
     *
     * @param input
     */
    void callImpl(const FunctionBase::InputRefVector &input) override {
        if (f_ == nullptr) {
            throw std::runtime_error(
                "Function callback not provided to CallbackFunction!");
        }
        f_(input, OutputVector());
    }

   private:
    f_callback_ f_ = nullptr;
};

}  // namespace common
}  // namespace damotion

#endif /* COMMON_FUNCTION_H */
