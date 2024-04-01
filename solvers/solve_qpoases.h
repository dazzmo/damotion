#ifndef CONTROL_SOLVE_QPOASES_H
#define CONTROL_SOLVE_QPOASES_H

#include <qpOASES.hpp>

#include "common/profiler.h"
#include "solvers/program.h"
#include "solvers/solver.h"

namespace damotion {
namespace optimisation {
namespace solvers {

class QPOASESSolverInstance : public SolverBase {
   public:
    QPOASESSolverInstance() = default;

    QPOASESSolverInstance(Program& prog) : SolverBase(prog) {
        // Create problem
        qp_ = std::make_unique<qpOASES::SQProblem>(
            GetCurrentProgram().NumberOfDecisionVariables(),
            GetCurrentProgram().NumberOfConstraints());
        lbx_ = Eigen::VectorXd::Zero(
            GetCurrentProgram().NumberOfDecisionVariables());
        ubx_ = Eigen::VectorXd::Zero(
            GetCurrentProgram().NumberOfDecisionVariables());

        // Create variable bounds from bounding box constraints
        for (Binding<BoundingBoxConstraint>& binding :
             GetCurrentProgram().GetBoundingBoxConstraintBindings()) {
            // For each variable of the constraint
            const sym::VariableVector& v = binding.GetVariable(0);
            for (int i = 0; i < v.size(); ++i) {
                lbx_[GetCurrentProgram().GetDecisionVariableIndex(v[i])] =
                    binding.Get().LowerBound()[i];
                ubx_[GetCurrentProgram().GetDecisionVariableIndex(v[i])] =
                    binding.Get().UpperBound()[i];
            }
        }

        // Constraint bounds
        ubA_ = Eigen::VectorXd::Zero(GetCurrentProgram().NumberOfConstraints());
        lbA_ = Eigen::VectorXd::Zero(GetCurrentProgram().NumberOfConstraints());

        std::cout << lbx_.transpose() << std::endl;
        std::cout << ubx_.transpose() << std::endl;
        std::cout << lbA_.transpose() << std::endl;
        std::cout << ubA_.transpose() << std::endl;

        H_ = Eigen::MatrixXd::Zero(
            GetCurrentProgram().NumberOfDecisionVariables(),
            GetCurrentProgram().NumberOfDecisionVariables());

        g_ = Eigen::VectorXd::Zero(
            GetCurrentProgram().NumberOfDecisionVariables());

        // Sparse Method

        // /* Construct sparse symmetric Hessian in qpOASES environment  */
        // const Eigen::SparseMatrix<double>& H =
        //     GetCurrentProgram().LagrangianHessian();

        // int* h_colind = const_cast<int*>(H.outerIndexPtr());
        // int* h_row = const_cast<int*>(H.innerIndexPtr());
        // double* h = const_cast<double*>(H.valuePtr());
        // // Get matrix data in CCS
        // H_ = std::make_unique<qpOASES::SymSparseMat>(H.rows(), H.cols(),
        // h_row,
        //                                              h_colind, h);
        // H_->createDiagInfo();
        // /* Construct sparse constraint Jacobian in qpOASES environment  */
        // const Eigen::SparseMatrix<double>& A =
        //     GetCurrentProgram().ConstraintJacobian();
        // int* a_colind = const_cast<int*>(A.outerIndexPtr());
        // int* a_row = const_cast<int*>(A.innerIndexPtr());
        // double* a = const_cast<double*>(A.valuePtr());

        // A_ = std::make_unique<qpOASES::SparseMatrix>(A.rows(), A.cols(),
        // a_row,
        //                                              a_colind, a);
    }

    ~QPOASESSolverInstance() {}

    void Solve() {
        common::Profiler profiler("QPOASESSolverInstance::Solve");
        // Number of decision variables in the program
        int nx = GetCurrentProgram().NumberOfDecisionVariables();
        int nc = GetCurrentProgram().NumberOfConstraints();

        // Linear costs
        std::cout << "Linear cost\n";
        for (Binding<LinearCost>& binding :
             GetCurrentProgram().GetLinearCostBindings()) {
            const std::vector<bool>& continuous =
                CostBindingContinuousInputCheck(binding);

            // Evaluate the cost
            EvaluateCost(binding.Get(), primal_solution_x_,
                         binding.GetVariables(), continuous, true, true, false);

            // Update the gradient
            for (int i = 0; i < binding.GetVariables().size(); ++i) {
                UpdateVectorAtVariableLocations(g_, binding.Get().c(),
                                                binding.GetVariable(i),
                                                continuous[i]);
            }
        }
        // Quadratic costs
        std::cout << "Quadratic cost\n";
        for (Binding<QuadraticCost>& binding :
             GetCurrentProgram().GetQuadraticCostBindings()) {
            const std::vector<bool>& continuous =
                CostBindingContinuousInputCheck(binding);

            // Evaluate the cost
            std::cout << "Eval\n";
            EvaluateCost(binding.Get(), primal_solution_x_,
                         binding.GetVariables(), continuous, true, true, false);

            std::cout << "Gradient\n";
            // Update the gradient
            for (int i = 0; i < binding.GetVariables().size(); ++i) {
                UpdateVectorAtVariableLocations(g_, binding.Get().g(),
                                                binding.GetVariable(i),
                                                continuous[i]);
            }
            std::cout << "Hessian\n";
            // Update the hessian
            for (int i = 0; i < binding.GetVariables().size(); ++i) {
                for (int j = i; j < binding.GetVariables().size(); ++j) {
                    UpdateHessianAtVariableLocations(
                        H_, 2.0 * binding.Get().Q(), binding.GetVariable(i),
                        binding.GetVariable(j), continuous[i], continuous[j]);
                }
            }
        }

        std::cout << "Linear constraints\n";
        // Evaluate only the linear constraints of the program
        int idx = 0;
        for (Binding<LinearConstraint>& binding :
             GetCurrentProgram().GetLinearConstraintBindings()) {
            // Compute the constraints
            EvaluateConstraint(
                binding.Get(), idx, primal_solution_x_, binding.GetVariables(),
                ConstraintBindingContinuousInputCheck(binding), true);

            // Adapt bounds for the linear constraints
            ubA_.middleRows(idx, binding.Get().Dimension()) =
                binding.Get().UpperBound() - binding.Get().b();
            lbA_.middleRows(idx, binding.Get().Dimension()) =
                binding.Get().LowerBound() - binding.Get().b();

            // Increase constraint index
            idx += binding.Get().Dimension();
        }

        std::cout << objective_cache_ << std::endl;
        std::cout << g_ << std::endl;
        std::cout << H_ << std::endl;

        // TODO - Map to row major
        typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>
            RowMajorMatrixXd;
        RowMajorMatrixXd H = Eigen::Map<RowMajorMatrixXd>(H_.data(), nx, nx);

        RowMajorMatrixXd A = Eigen::Map<RowMajorMatrixXd>(
            constraint_jacobian_cache_.data(), nc, nx);

        // Solve
        int nWSR = 100;
        if (n_solves_ == 0) {
            qp_->init(H.data(), objective_gradient_cache_.data(), A.data(),
                      lbx_.data(), ubx_.data(), lbA_.data(), ubA_.data(), nWSR);
        } else {
            qp_->hotstart(H.data(), objective_gradient_cache_.data(), A.data(),
                          lbx_.data(), ubx_.data(), lbA_.data(), ubA_.data(),
                          nWSR);
        }

        // Get primal solution
        qp_->getPrimalSolution(primal_solution_x_.data());
        std::cout << primal_solution_x_.transpose() << std::endl;

        // TODO: Handle Error
        n_solves_++;
    }

   private:
    int n_solves_ = 0;

    // Sparse method
    // std::unique_ptr<qpOASES::SymSparseMat> H_;
    // std::unique_ptr<qpOASES::SparseMatrix> A_;
    std::unique_ptr<qpOASES::SQProblem> qp_;

    Eigen::MatrixXd H_;
    Eigen::VectorXd g_;

    Eigen::VectorXd lbx_;
    Eigen::VectorXd ubx_;

    Eigen::VectorXd lbA_;
    Eigen::VectorXd ubA_;
};

class QPOASESSolver {
   public:
    QPOASESSolver() = default;
    ~QPOASESSolver() = default;

    QPOASESSolver(Program& program) {
        // Create new instance
        qp_ = std::make_unique<QPOASESSolverInstance>(program);
    }

    // Update program // TODO - Throw warning if program sizes don't match?
    void UpdateProgram(Program& program) { qp_->UpdateProgram(program); }

    void Solve() { qp_->Solve(); }

    const Eigen::VectorXd& GetPrimalSolution() const {
        return qp_->GetPrimalSolution();
    }

   private:
    // Constraint bindings
    std::unique_ptr<QPOASESSolverInstance> qp_;
};

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVE_QPOASES_H */
