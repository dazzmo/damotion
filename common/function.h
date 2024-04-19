#ifndef COMMON_FUNCTION_H
#define COMMON_FUNCTION_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <functional>
#include <iostream>

#include "common/logging.h"

namespace damotion {
namespace common {

/**
 * @brief Vector of input vector references to the function
 *
 */
typedef std::vector<Eigen::Ref<const Eigen::VectorXd>> InputRefVector;

template <typename MatrixType>
class Function {
   public:
    Function() : n_in_(0), n_out_(0) {}

    Function(const int n_in, const int n_out) : n_in_(n_in), n_out_(n_out) {}

    ~Function() = default;

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

    /**
     * @brief Update the function based on its inputs
     *
     * @param input List of input vectors for the function
     * @param check Perform checks on the input to ensure correct size and good
     * data
     */
    void call(const InputRefVector &input, bool check = false) {
        if (check) CheckInputRefVector(input);
        callImpl(input);
    }

    /**
     * @brief Returns the current value of output i
     *
     * @param i
     * @return const MatrixType&
     */
    const MatrixType &getOutput(int i) const { return out_[i]; }

   protected:
    void SetNumberOfInputs(const int &n) { n_in_ = n; }
    void SetNumberOfOutputs(const int &n) { n_out_ = n; }

    /**
     * @brief Virtual method for derived class to override
     *
     * @param input
     */
    virtual void callImpl(const InputRefVector &input) = 0;

    /**
     * @brief Vector of output matrices
     *
     * @return std::vector<Eigen::MatrixXd>&
     */
    std::vector<MatrixType> &OutputVector() { return out_; }

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

    std::vector<MatrixType> out_;
};

/**
 * @brief Function that operates by callback
 *
 */
template <typename MatrixType>
class CallbackFunction : public Function<MatrixType> {
   public:
    typedef std::function<void(const InputRefVector &,
                               std::vector<MatrixType> &)>
        f_callback_;

    CallbackFunction() = default;
    ~CallbackFunction() = default;

    CallbackFunction(const int n_in, const int n_out,
                     const f_callback_ &callback)
        : Function<MatrixType>(n_in, n_out) {
        SetCallback(callback);
    }

    /**
     * @brief Initialise the output i with the values given by val
     *
     * @param i
     * @param val
     */
    void InitOutput(const int i, const MatrixType &val) {
        this->OutputVector()[i] = val;
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
    void callImpl(const InputRefVector &input) override {
        if (f_ == nullptr) {
            throw std::runtime_error(
                "Function callback not provided to CallbackFunction!");
        }
        f_(input, this->OutputVector());
    }

   private:
    f_callback_ f_ = nullptr;
};

}  // namespace common
}  // namespace damotion

#endif /* COMMON_FUNCTION_H */
