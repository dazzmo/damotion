#ifndef COMMON_FUNCTION_H
#define COMMON_FUNCTION_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <functional>
#include <iostream>

namespace damotion {
namespace common {

class Function {
   public:
    /**
     * @brief Vector of input vector references to the function
     *
     */
    typedef std::vector<Eigen::Ref<const Eigen::VectorXd>> InputRefVector;

    Function() : n_in_(0), n_out_(0), out_({}) {}

    Function(const int n_in, const int n_out)
        : n_in_(n_in), n_out_(n_out), out_(n_out) {}

    Function(const Function &other);
    Function &operator=(const Function &other);

    ~Function();

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
     * @brief Pure virtual method that updates the function based on its inputs
     *
     * @param List of input vectors for the function
     * @param check Perform checks on the input to ensure correct size and good
     * data
     */
    void call(const InputRefVector &input, bool check = false) {
        if (check) CheckInputRefVector(input);
        callImpl(input);
    }

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
    const Eigen::Ref<const Eigen::MatrixXd> getOutput(int i);

    /**
     * @brief Returns the sparse matrix output i. You must call
     * setSparseOutput(i) beforehand to return a sparse output for output i.
     *
     * @param i
     * @return const Eigen::SparseMatrix<double>&
     */
    const Eigen::Ref<const Eigen::SparseMatrix<double>> getOutputSparse(int i);

   protected:
    // Dense matrix outputs
    std::vector<Eigen::MatrixXd> out_;
    // Sparse matrix outputs
    std::vector<Eigen::SparseMatrix<double>> out_sparse_;

    // Flag to indicate if output has been set as sparse
    std::vector<bool> is_out_sparse_;

    void SetNumberOfInputs(const int &n) { n_in_ = n; }
    void SetNumberOfOutputs(const int &n) { n_out_ = n; }

    /**
     * @brief Virtual call method for derived class to override
     *
     * @param input
     */
    virtual void callImpl(const InputRefVector &input) = 0;

   private:
    int n_in_;
    int n_out_;

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
};

/**
 * @brief Function that operates by callback
 *
 */
class CallbackFunction : public Function {
   public:
    typedef std::function<void(const Function::InputRefVector &,
                               std::vector<Eigen::MatrixXd> &)>
        f_callback_;

    CallbackFunction() = default;
    ~CallbackFunction() = default;

    CallbackFunction(const int n_in, const int n_out,
                     const f_callback_ &callback)
        : Function(n_in, n_out) {
        SetCallback(callback);
    }

    void SetCallback(const f_callback_ &callback) { f_ = callback; }

    void callImpl(const Function::InputRefVector &input) override {
        std::cout << "callImpl\n";
        f_(input, out_);
    }

   private:
    f_callback_ f_;
};

}  // namespace common
}  // namespace damotion

#endif /* COMMON_FUNCTION_H */
