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

        std::cout << lbx_ << std::endl;
        std::cout << ubx_ << std::endl;

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

        std::cout << lbx_ << std::endl;
        std::cout << ubx_ << std::endl;

        // Constraint bounds
        ubA_ = Eigen::VectorXd::Zero(GetCurrentProgram().NumberOfConstraints());
        lbA_ = Eigen::VectorXd::Zero(GetCurrentProgram().NumberOfConstraints());

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

        // Evaluate costs
        EvaluateCosts(primal_solution_x_, true, true);

        // Double the values of the Hessian to accomodate for the quadratic
        // form 0.5 x^T Q x + g^T x
        lagrangian_hes_cache_ *= 2.0;

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

        // TODO - Map to row major

        // Solve
        int nWSR = 100;
        // if (n_solves_ == 0) {
        //     qp_->init(lagrangian_hes_cache_.data(),
        //               objective_gradient_cache_.data(),
        //               constraint_jacobian_cache_.data(),
        //               GetCurrentProgram().DecisionVariablesLowerBound().data(),
        //               GetCurrentProgram().DecisionVariablesUpperBound().data(),
        //               lbA.data(), ubA.data(), nWSR);
        // } else {
        //     qp_->hotstart(
        //         lagrangian_hes_cache_.data(),
        //         objective_gradient_cache_.data(),
        //         constraint_jacobian_cache_.data(),
        //         GetCurrentProgram().DecisionVariablesLowerBound().data(),
        //         GetCurrentProgram().DecisionVariablesUpperBound().data(),
        //         lbA.data(), ubA.data(), nWSR);
        // }

        // Get primal solution
        // qp_->getPrimalSolution(primal_solution_x_.data());

        // TODO: Handle Error
        n_solves_++;
    }

   private:
    int n_solves_ = 0;

    // Sparse method
    std::unique_ptr<qpOASES::SymSparseMat> H_;
    std::unique_ptr<qpOASES::SparseMatrix> A_;
    std::unique_ptr<qpOASES::SQProblem> qp_;

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
