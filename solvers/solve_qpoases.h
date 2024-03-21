#ifndef CONTROL_SOLVE_QPOASES_H
#define CONTROL_SOLVE_QPOASES_H
#ifndef SOLVE_QPOASES_H
#define SOLVE_QPOASES_H

#include <qpOASES.hpp>

#include "solvers/program.h"
#include "solvers/solver.h"

namespace damotion {
namespace solvers {

class QPOASESSolverInstance : public SolverBase {
   public:
    QPOASESSolverInstance(Program& prog) : SolverBase(prog) {
        // Create problem
        qp_ = std::make_unique<qpOASES::SQProblem>(
            GetCurrentProgram().NumberOfDecisionVariables(),
            GetCurrentProgram().NumberOfConstraints());

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
        // Number of decision variables in the program
        int n = GetCurrentProgram().NumberOfDecisionVariables();

        // Dummy decision variable input
        Eigen::VectorXd x(n);
        x.setZero();

        EvaluateCosts(x, true, true);

        // Double the values of the Hessian to accomodate for the quadratic form
        // 0.5 x^T Q x + g^T x
        lagrangian_hes_cache_ *= 2.0;

        // Evaluate constraints and Jacobian
        EvaluateConstraints(x, true);

        // TODO - Map to row major

        // Solve
        int nWSR = 100;
        if (n_solves_ == 0) {
            qp_->init(lagrangian_hes_cache_.data(),
                      objective_gradient_cache_.data(),
                      constraint_jacobian_cache_.data(),
                      GetCurrentProgram().DecisionVariablesLowerBound().data(),
                      GetCurrentProgram().DecisionVariablesUpperBound().data(),
                      GetCurrentProgram().ConstraintsLowerBound().data(),
                      GetCurrentProgram().ConstraintsUpperBound().data(), nWSR);
        } else {
            // qp_->hotstart(
            //     H_.get(), objective_gradient_cache_.data(), A_.get(),
            //     GetCurrentProgram().DecisionVariablesLowerBound().data(),
            //     GetCurrentProgram().DecisionVariablesUpperBound().data(),
            //     GetCurrentProgram().ConstraintsLowerBound().data(),
            //     GetCurrentProgram().ConstraintsUpperBound().data(), nWSR);
        }

        // Get primal solution
        qp_->getPrimalSolution(primal_solution_x_.data());

        // TODO: Handle Error
        n_solves_++;
    }

   private:
    int n_solves_ = 0;

    // Sparse method
    std::unique_ptr<qpOASES::SymSparseMat> H_;
    std::unique_ptr<qpOASES::SparseMatrix> A_;
    std::unique_ptr<qpOASES::SQProblem> qp_;
};

class QPOASESSolver {
   public:
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
    std::unique_ptr<QPOASESSolverInstance> qp_;
};

}  // namespace solvers
}  // namespace damotion
#endif /* SOLVE_QPOASES_H */

#endif /* CONTROL_SOLVE_QPOASES_H */