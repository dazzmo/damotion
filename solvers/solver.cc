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

void SolverBase::EvaluateCost(Cost& cost, const Eigen::VectorXd& x,
                              const std::vector<sym::VariableVector>& var,
                              const std::vector<const double*>& par,
                              const std::vector<bool>& continuous, bool grd,
                              bool hes, bool update_cache) {
    int nv = var.size();
    int np = par.size();
    // Create inputs to evaluate the function
    std::vector<const double*> inputs(nv, nullptr);
    // Optional creation of vectors for inputs to the functions (if vector input
    // is not continuous in optimisation vector)
    std::vector<Eigen::VectorXd> vecs = {};

    // Determine inputs to the function
    for (int i = 0; i < nv; ++i) {
        if (continuous[i]) {
            // Set input to the start of the vector input
            inputs[i] = x.data() +
                        GetCurrentProgram().GetDecisionVariableIndex(var[i][0]);
        } else {
            // Construct a vector for this input
            Eigen::VectorXd xi(var[i].size());
            for (int ii = 0; ii < var[i].size(); ++ii) {
                xi[ii] =
                    x[GetCurrentProgram().GetDecisionVariableIndex(var[i][ii])];
            }
            vecs.push_back(xi);
            inputs[i] = vecs.back().data();
        }
    }

    // Set variables
    for (int i = 0; i < nv; ++i) {
        cost.ObjectiveFunction().setInput(i, inputs[i]);
        if (grd) {
            if (!cost.HasGradient()) {
                throw std::runtime_error("Cost does not have a Gradient!");
            }
            cost.GradientFunction().setInput(i, inputs[i]);
        }
        if (hes) {
            if (!cost.HasHessian()) {
                throw std::runtime_error("Cost does not have a Hessian!");
            }
            cost.HessianFunction().setInput(i, inputs[i]);
        }
    }
    // Set parameters
    for (int i = 0; i < np; ++i) {
        cost.ObjectiveFunction().setInput(nv + i, par[i]);
        if (grd) {
            if (!cost.HasGradient()) {
                throw std::runtime_error("Cost does not have a Gradient!");
            }
            cost.GradientFunction().setInput(nv + i, par[i]);
        }
        if (hes) {
            if (!cost.HasHessian()) {
                throw std::runtime_error("Cost does not have a Hessian!");
            }
            cost.HessianFunction().setInput(nv + i, par[i]);
        }
    }

    cost.ObjectiveFunction().call();
    if (grd) cost.GradientFunction().call();
    if (hes) cost.HessianFunction().call();

    if (update_cache == false) return;

    objective_cache_ += cost.ObjectiveFunction().getOutput(0).data()[0];

    if (grd) {
        for (int i = 0; i < nv; ++i) {
            UpdateVectorAtVariableLocations(
                objective_gradient_cache_, cost.GradientFunction().getOutput(i),
                var[i], continuous[i]);
        }
    }

    if (hes) {
        int cnt = 0;
        for (int i = 0; i < nv; ++i) {
            for (int j = i; j < nv; ++j) {
                // Increase the count for the Hessian
                UpdateHessianAtVariableLocations(
                    lagrangian_hes_cache_,
                    cost.HessianFunction().getOutput(cnt), var[i], var[j],
                    continuous[i], continuous[j]);
                cnt++;
            }
        }
    }
}

void SolverBase::EvaluateCosts(const Eigen::VectorXd& x, bool grad, bool hes) {
    // Reset objective
    objective_cache_ = 0.0;
    // Reset objective gradient
    if (grad) {
        objective_gradient_cache_.setZero();
    }
    if (hes) {
        lagrangian_hes_cache_.setZero();
    }
    // // Evaluate all costs in the program
    // for (Binding<Cost>& b : prog_.GetAllCostBindings()) {
    //     EvaluateCost(b, x, grad, hes);
    // }
}

// Evaluates the constraint and updates the cache for the gradients
void SolverBase::EvaluateConstraint(Constraint& c, const int& constraint_idx,
                                    const Eigen::VectorXd& x,
                                    const std::vector<sym::VariableVector>& var,
                                    const std::vector<const double*>& par,
                                    const std::vector<bool>& continuous,
                                    bool jac, bool update_cache) {
    // Get size of constraint
    int nc = c.Dimension();
    int nv = var.size();
    int np = par.size();

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
            inputs[i] =
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
    for (int i = 0; i < nv; ++i) {
        c.ConstraintFunction().setInput(i, inputs[i]);
        if (jac) c.JacobianFunction().setInput(i, inputs[i]);
    }
    // Set parameter inputs
    for (int i = 0; i < np; ++i) {
        c.ConstraintFunction().setInput(nv + i, par[i]);
        if (jac) c.JacobianFunction().setInput(nv + i, par[i]);
    }

    c.ConstraintFunction().call();
    if (jac) {
        if (!c.HasJacobian()) {
            throw std::runtime_error("Constraint " + c.name() +
                                     " does not have a Jacobian!");
        }
        c.JacobianFunction().call();
    }

    if (update_cache == false) return;

    constraint_cache_.middleRows(constraint_idx, nc) =
        c.ConstraintFunction().getOutput(0);

    // ! Currently a dense jacobian - Look into sparse work
    // Update Jacobian blocks
    for (int i = 0; i < nv; ++i) {
        UpdateJacobianAtVariableLocations(
            constraint_jacobian_cache_, constraint_idx,
            c.JacobianFunction().getOutput(i), var[i], continuous[i]);
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
    for (auto& b : program.GetAllCostBindings()) {
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

// TODO - Make utility for block-filling
void SolverBase::UpdateVectorAtVariableLocations(Eigen::VectorXd& res,
                                                 const Eigen::VectorXd& block,
                                                 const sym::VariableVector& var,
                                                 bool is_continuous) {
    if (is_continuous) {
        res.middleRows(GetCurrentProgram().GetDecisionVariableIndex(var[0]),
                       var.size()) += block;
    } else {
        // For each variable, update the location in the Jacobian
        for (int j = 0; j < var.size(); ++j) {
            int idx = GetCurrentProgram().GetDecisionVariableIndex(var[j]);
            res[idx] += block[j];
        }
    }
}

void SolverBase::UpdateJacobianAtVariableLocations(
    Eigen::MatrixXd& jac, int row_idx, const Eigen::MatrixXd& block,
    const sym::VariableVector& var, bool is_continuous) {
    Eigen::Block<Eigen::MatrixXd> J = jac.middleRows(row_idx, block.rows());
    if (is_continuous) {
        J.middleCols(GetCurrentProgram().GetDecisionVariableIndex(var[0]),
                     var.size()) = block;
    } else {
        // For each variable, update the location in the Jacobian
        for (int j = 0; j < var.size(); ++j) {
            int idx = GetCurrentProgram().GetDecisionVariableIndex(var[j]);
            J.col(idx) = block.col(j);
        }
    }
}

void SolverBase::UpdateHessianAtVariableLocations(
    Eigen::MatrixXd& hes, const Eigen::MatrixXd& block,
    const sym::VariableVector& var_x, const sym::VariableVector& var_y,
    bool is_continuous_x, bool is_continuous_y) {
    // For each variable combination
    if (is_continuous_x && is_continuous_y) {
        int i_idx = GetCurrentProgram().GetDecisionVariableIndex(var_x[0]),
            j_idx = GetCurrentProgram().GetDecisionVariableIndex(var_y[0]);
        int i_sz = var_x.size(), j_sz = var_y.size();
        // Create lower triangular Hessian
        if (i_idx > j_idx) {
            // Block fill
            hes.block(i_idx, j_idx, i_sz, j_sz) += block;
        } else {
            hes.block(j_idx, i_idx, j_sz, i_sz) += block.transpose();
        }

    } else {
        // For each variable pair, populate the Hessian
        for (int ii = 0; ii < var_x.size(); ++ii) {
            for (int jj = 0; jj < var_y.size(); ++jj) {
                int i_idx =
                        GetCurrentProgram().GetDecisionVariableIndex(var_x[ii]),
                    j_idx =
                        GetCurrentProgram().GetDecisionVariableIndex(var_y[jj]);
                // Create lower triangular matrix
                if (i_idx > j_idx) {
                    hes(i_idx, j_idx) += block(ii, jj);
                } else {
                    hes(j_idx, i_idx) += block(ii, jj);
                }
            }
        }
    }
}

}  // namespace solvers
}  // namespace optimisation
}  // namespace damotion