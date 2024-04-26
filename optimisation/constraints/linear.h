#ifndef CONSTRAINTS_LINEAR_H
#define CONSTRAINTS_LINEAR_H

#include "optimisation/constraints/base.h"

namespace damotion {
namespace optimisation {

template <typename MatrixType>
class LinearConstraint : public ConstraintBase<MatrixType> {
   public:
    /**
     * @brief Construct a new Linear Constraint object of the form \f$ A x + b
     * \f$
     *
     * @param name Name of the constraint. Default name given if provided ""
     * @param A Vector of coefficient matrices
     * @param b
     * @param p Parameters that A and b depend on
     * @param bounds
     * @param jac
     */
    LinearConstraint(const std::string &name, const casadi::SX &A,
                     const casadi::SX &b, const casadi::SXVector &p,
                     const BoundsType &bounds, bool jac = true)
        : ConstraintBase<MatrixType>(name, "linear_constraint") {
        ConstructConstraint(A, b, p, bounds, jac);
    }

    LinearConstraint(const std::string &name, const Eigen::MatrixXd &A,
                     const Eigen::VectorXd &b, const BoundsType &bounds,
                     bool jac = true)
        : ConstraintBase<MatrixType>(name, "linear_constraint") {
        // Constant vector b
        casadi::DM Ad, bd;
        damotion::utils::casadi::toCasadi(b, bd);
        damotion::utils::casadi::toCasadi(A, Ad);
        casadi::SX bsx = bd;
        casadi::SX Asx = Ad;

        // Construct constraint
        ConstructConstraint(Asx, bsx, {}, bounds, jac);
    }

    LinearConstraint(const std::string &name, const sym::Expression &ex,
                     const BoundsType &bounds, bool jac = true)
        : ConstraintBase<MatrixType>(name, "linear_constraint") {
        // Extract linear form
        casadi::SX A, b;
        casadi::SX::linear_coeff(ex, ex.Variables()[0], A, b, true);

        ConstructConstraint(A, b, ex.Parameters(), bounds, jac);
    }

    void SetCallback(
        const typename common::CallbackFunction<MatrixType>::f_callback_ &fA,
        const typename common::CallbackFunction<Eigen::VectorXd>::f_callback_
            &fb) {
        fA_ = std::make_shared<common::CallbackFunction<MatrixType>>(1, 1, fA);
        fb_ = std::make_shared<common::CallbackFunction<Eigen::VectorXd>>(1, 1,
                                                                          fb);
    }

    /**
     * @brief The coefficient matrix A for the expression A x + b.
     *
     * @return const Eigen::MatrixXd&
     */
    const Eigen::Ref<const Eigen::MatrixXd> A() { return fA_->getOutput(0); }

    /**
     * @brief The constant vector for the linear constraint A x + b
     *
     * @return const Eigen::VectorXd&
     */
    const Eigen::Ref<const Eigen::VectorXd> b() { return fb_->getOutput(0); }

    void eval(const common::InputRefVector &in, bool jac = true,
              bool hes = true) override {
        // Evaluate the coefficients
        common::InputRefVector in_Ab = {};
        for (int i = 1; i < input.size(); ++i) {
            in_Ab.push_back(input[i]);
        }

        // Evaluate the constraint
        Eigen::VectorXd c = fA_->getOutput(0) * in[0] + fb_->getOutput(0);
        if (jac) MatrixType J = fA_->getOutput(0);
    }

   private:
    std::shared_ptr<common::Function<MatrixType>> fA_;
    std::shared_ptr<common::Function<Eigen::VectorXd>> fb_;

    void ConstructConstraint(const casadi::SX &A, const casadi::SX &b,
                             const casadi::SXVector &p,
                             const BoundsType &bounds, bool jac = true,
                             bool sparse = false) {
        casadi::SXVector in = {};
        int nvar = 0;
        assert(A.rows() == b.rows() && "A and b must be same dimension!");

        // Create constraint dimensions and update bounds
        this->Resize(b.rows(), in.size(), p.size());
        this->UpdateBounds(bounds);

        // Add any parameters that define A and b
        for (const casadi::SX &pi : p) {
            in.push_back(pi);
        }

        if (std::is_same<MatrixType, Eigen::SparseMatrix<double>>::value) {
            fA_ = std::make_shared<utils::casadi::FunctionWrapper<MatrixType>>(
                casadi::Function(this->name() + "_A", in, {A}));
        } else {
            fA_ = std::make_shared<utils::casadi::FunctionWrapper<MatrixType>>(
                casadi::Function(this->name() + "_A", in, {densify(A)}));
        }

        fb_ = std::make_shared<utils::casadi::FunctionWrapper<Eigen::VectorXd>>(
            casadi::Function(this->name() + "_b", in, {densify(b)}));
    }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* CONSTRAINTS_LINEAR_H */
