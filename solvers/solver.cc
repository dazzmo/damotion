#include "solvers/solver.h"

namespace damotion {
namespace optimisation {
namespace solvers {

SolverBase::SolverBase(Program& prog) : prog_(prog) {
    // Construct caches
    // ! Program is currently dense, look at sparse alternative soon

    // Decision variables
    decision_variable_cache_ =
        Eigen::VectorXd::Zero(prog_.NumberOfDecisionVariables());
    // Output solution vector
    primal_solution_x_ =
        Eigen::VectorXd::Zero(prog_.NumberOfDecisionVariables());
    // Objective gradient
    objective_gradient_cache_ =
        Eigen::VectorXd::Zero(prog_.NumberOfDecisionVariables());
    // Constraint vector
    constraint_cache_ = Eigen::VectorXd::Zero(prog_.NumberOfConstraints());
    // Dual variables
    dual_variable_cache_ = Eigen::VectorXd::Zero(prog_.NumberOfConstraints());
    // Dense constraint Jacobian
    constraint_jacobian_cache_ = Eigen::MatrixXd::Zero(
        prog_.NumberOfConstraints(), prog_.NumberOfDecisionVariables());

    // Dense constraint Hessian
    lagrangian_hes_cache_ = Eigen::MatrixXd::Zero(
        prog_.NumberOfDecisionVariables(), prog_.NumberOfDecisionVariables());

    CalculateBindingInputs();
}

void SolverBase::EvaluateCost(Binding<Cost>& b, const Eigen::VectorXd& x,
                              bool grd, bool hes) {
    Cost& cost = b.Get();

    // Get data for continuous memory
    // TODO - Check it exists
    std::vector<bool>& is_continuous = cost_binding_continuous_input_[b.id()];

    // Create inputs to evaluate the function
    std::vector<const double*> inputs(b.NumberOfVariables(), nullptr);
    // Optional creation of vectors for inputs to the functions (if vector input
    // is not continuous in optimisation vector)
    std::vector<Eigen::VectorXd> vecs = {};

    // Determine inputs to the function
    for (int i = 0; i < b.NumberOfVariables(); ++i) {
        if (is_continuous[i]) {
            // Set input to the start of the vector input
            inputs[0] = x.data() + GetCurrentProgram().GetDecisionVariableIndex(
                                       b.GetVariable(i)[0]);
        } else {
            // Construct a vector for this input
            Eigen::VectorXd xi(b.GetVariable(i).size());
            for (int ii = 0; ii < b.GetVariable(i).size(); ++ii) {
                xi[ii] = x[GetCurrentProgram().GetDecisionVariableIndex(
                    b.GetVariable(i)[ii])];
            }
            vecs.push_back(xi);
            inputs[i] = xi.data();
        }
    }

    for (int i = 0; i < inputs.size(); ++i) {
        cost.ObjectiveFunction().setInput(i, inputs[i]);
        if (grd) cost.GradientFunction().setInput(i, inputs[i]);
        if (hes) cost.HessianFunction().setInput(i, inputs[i]);
    }

    cost.ObjectiveFunction().call();
    objective_cache_ += cost.ObjectiveFunction().getOutput(0).data()[0];

    if (grd) {
        if (!cost.HasGradient()) {
            throw std::runtime_error("Cost does not have a Gradient!");
        }

        // Evaluate the jacobians
        cost.GradientFunction().call();

        for (int i = 0; i < b.NumberOfVariables(); ++i) {
            if (is_continuous[i]) {
                objective_gradient_cache_.middleRows(
                    GetCurrentProgram().GetDecisionVariableIndex(
                        b.GetVariable(i)[0]),
                    b.GetVariable(i).size()) +=
                    cost.GradientFunction().getOutput(i);
            } else {
                // For each variable, update the location in the Jacobian
                for (int j = 0; j < b.GetVariable(i).size(); ++j) {
                    int idx = GetCurrentProgram().GetDecisionVariableIndex(
                        b.GetVariable(i)[j]);
                    objective_gradient_cache_[idx] +=
                        cost.GradientFunction().getOutput(i).data()[j];
                }
            }
        }
    }

    if (hes) {
        if (!cost.HasHessian()) {
            throw std::runtime_error("Cost does not have a Hessian!");
        }
        // Evaluate the hessians
        cost.HessianFunction().call();

        int cnt = 0;
        std::cout << b.NumberOfVariables() << std::endl;
        for (int i = 0; i < b.NumberOfVariables(); ++i) {
            for (int j = i; j < b.NumberOfVariables(); ++j) {
                // For each variable combination
                if (is_continuous[i] && is_continuous[j]) {
                    // Block fill
                    lagrangian_hes_cache_.block(
                        GetCurrentProgram().GetDecisionVariableIndex(
                            b.GetVariable(i)[0]),
                        GetCurrentProgram().GetDecisionVariableIndex(
                            b.GetVariable(j)[0]),
                        b.GetVariable(i).size(), b.GetVariable(j).size()) +=
                        cost.HessianFunction().getOutput(cnt);
                } else {
                    // For each variable pair, populate the Hessian
                    for (int ii = 0; ii < b.GetVariable(i).size(); ++ii) {
                        for (int jj = 0; jj < b.GetVariable(j).size(); ++jj) {
                            lagrangian_hes_cache_(
                                GetCurrentProgram().GetDecisionVariableIndex(
                                    b.GetVariable(i)[ii]),
                                GetCurrentProgram().GetDecisionVariableIndex(
                                    b.GetVariable(j)[jj])) +=
                                cost.HessianFunction().getOutput(cnt)(ii, jj);
                        }
                    }
                }
            }
        }
    }
}

// TODO - Make utility for block-filling

void SolverBase::EvaluateCosts(const Eigen::VectorXd& x, bool grad, bool hes) {
    // Reset objective
    objective_cache_ = 0.0;
    // Reset objective gradient
    if (grad) {
        objective_gradient_cache_.setZero();
    }
    // Evaluate all costs in the program
    for (Binding<Cost>& b : prog_.GetCostBindings()) {
        EvaluateCost(b, x, grad, hes);
    }
}

// Evaluates the constraint and updates the cache for the gradients
void SolverBase::EvaluateConstraint(Constraint& c, const int& constraint_idx,
                                    const Eigen::VectorXd& x,
                                    const std::vector<sym::VariableVector>& var,
                                    const std::vector<bool>& continuous,
                                    bool jac) {
    // Get size of constraint
    int nc = c.Dimension();
    int nv = var.size();

    // Create inputs to evaluate the function
    std::vector<const double*> inputs(nv, nullptr);
    // Optional creation of vectors for inputs to the functions (if vector
    // input is not continuous in optimisation vector)
    std::vector<Eigen::VectorXd> vecs = {};

    // Determine inputs to the function
    for (int i = 0; i < nv; ++i) {
        const sym::VariableVector& v = var[i];
        if (continuous[i]) {
            // Set input to the start of the vector input
            inputs[0] =
                x.data() + GetCurrentProgram().GetDecisionVariableIndex(v[0]);
        } else {
            // Construct a vector for this input
            Eigen::VectorXd xi(var.size());
            for (int j = 0; j < var.size(); ++j) {
                xi[j] = x[GetCurrentProgram().GetDecisionVariableIndex(v[j])];
            }
            vecs.push_back(xi);
            inputs[i] = xi.data();
        }
    }

    // Set vector inputs and evaluate necessary functions
    for (int i = 0; i < inputs.size(); ++i) {
        c.ConstraintFunction().setInput(i, inputs[i]);
        if (jac) c.JacobianFunction().setInput(i, inputs[i]);
    }
    c.ConstraintFunction().call();

    constraint_cache_.middleRows(constraint_idx, nc) =
        c.ConstraintFunction().getOutput(0);

    if (jac) {
        if (!c.HasJacobian()) {
            throw std::runtime_error("Constraint " + c.name() +
                                     " does not have a Jacobian!");
        }

        // Evaluate the jacobians
        c.JacobianFunction().call();

        // ! Currently a dense jacobian - Look into sparse work
        // Update Jacobian blocks
        Eigen::Block<Eigen::MatrixXd> J =
            constraint_jacobian_cache_.middleRows(constraint_idx, nc);

        for (int i = 0; i < nv; ++i) {
            const sym::VariableVector& v = var[i];

            if (continuous[i]) {
                J.middleCols(GetCurrentProgram().GetDecisionVariableIndex(v[0]),
                             var.size()) = c.JacobianFunction().getOutput(i);
            } else {
                // For each variable, update the location in the Jacobian
                for (int j = 0; j < var.size(); ++j) {
                    int idx =
                        GetCurrentProgram().GetDecisionVariableIndex(v[j]);
                    J.col(idx) = c.JacobianFunction().getOutput(i).col(j);
                }
            }
        }
    }
}

void SolverBase::EvaluateConstraints(const Eigen::VectorXd& x, bool jac) {
    // Reset constraint vector
    constraint_cache_.setZero();
    // Reset objective gradient
    if (jac) {
        constraint_jacobian_cache_.setZero();
    }
    // Loop through all constraints
}

void SolverBase::CalculateBindingInputs() {
    Program& program = GetCurrentProgram();
    // Pre-compute variable look ups for speed
    int id = 0;
    for (auto& b : program.GetAllConstraints()) {
        // For each input, assess if memory is contiguous (allow for
        // optimisation of input)
        std::vector<bool> is_continuous(b.NumberOfVariables());
        for (int i = 0; i < b.NumberOfVariables(); ++i) {
            if (IsContiguousInDecisionVariableVector(b.GetVariable(i))) {
                is_continuous[i] = true;
            } else {
                is_continuous[i] = false;
            }
        }
        // Add index to map
        constraint_binding_idx[b.id()] =
            constraint_binding_continuous_input_.size();
        // Include continuous data
        constraint_binding_continuous_input_.push_back(is_continuous);
    }

    // Perform the same for the costs
    for (auto& b : program.GetCostBindings()) {
        // For each input, assess if memory is contiguous (allow for
        // optimisation of input)
        std::vector<bool> is_continuous(b.NumberOfVariables());
        for (int i = 0; i < b.NumberOfVariables(); ++i) {
            if (IsContiguousInDecisionVariableVector(b.GetVariable(i))) {
                is_continuous[i] = true;
            } else {
                is_continuous[i] = false;
            }
        }
        // Add index to map
        cost_binding_idx[b.id()] = cost_binding_continuous_input_.size();
        // Include continuous data
        cost_binding_continuous_input_.push_back(is_continuous);
    }
}

bool SolverBase::IsContiguousInDecisionVariableVector(
    const sym::VariableVector& var) {
    Program& program = GetCurrentProgram();
    int idx = program.GetDecisionVariableIndex(var[0]);
    for (int i = 1; i < var.size(); ++i) {
        if (program.GetDecisionVariableIndex(var[i]) - idx != 1) {
            return false;
        }
        idx = program.GetDecisionVariableIndex(var[i]);
    }
    // Variables are continuous in the optimisation vector
    return true;
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion