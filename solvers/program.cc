#include "solvers/program.h"

namespace damotion {
namespace optimisation {

inline bool Program::IsDecisionVariable(const std::string &name) {
    // Indicate if it is present in the variable map
    return variables_id_.find(name) != variables_id_.end();
}

void Program::AddDecisionVariables(const std::string &name, const int n,
                                   const int m) {
    if (!IsDecisionVariable(name)) {
        variables_id_[name] = variables_.size();
        variables_.push_back(Variable(name, n, m));
    } else {
        std::cout << "Variables with name " << name
                  << " already exist in program!\n";
    }
}

void Program::SetDecisionVariablesFromVector(const Eigen::VectorXd &x) {
    damotion::common::Profiler profiler("Program::UpdateDecisionVariables");
    // Use current mapping to compute variables
    for (auto &xi : variables_id_) {
        int id = xi.second;
        Variable &v = variables_[id];
        v.val() << x.middleRows(variables_idx_[id], v.rows() * v.cols());
    }
}

void Program::ResizeDecisionVariables(const std::string &name, const int n,
                                      const int m) {
    int id = GetDecisionVariablesId(name);
    assert(id >= 0 && "Decision variable does not exist");
    variables_[id] = Variable(name, n, m);
}

int Program::GetDecisionVariablesId(const std::string &name) {
    auto p = variables_id_.find(name);
    if (p != variables_id_.end()) {
        return p->second;
    } else {
        return -1;
    }
}

Variable &Program::GetDecisionVariables(const std::string &name) {
    int id = GetDecisionVariablesId(name);
    assert(id >= 0 && "Variable not found in program!\n");
    return variables_[id];
}

void Program::RemoveDecisionVariables(const std::string &name) {
    // Delete the entry and re-id other variables
    int id = GetDecisionVariablesId(name);
    assert(id >= 0 && "Variable not found in program!\n");
    // Delete variable
    variables_id_.erase(name);
    variables_.erase(std::next(variables_.begin() + id));
    // Re-id remaining variables
    for (int i = 0; i < variables_.size(); ++i) {
        variables_id_[variables_[i].name()] = i;
    }
}

void Program::UpdateDecisionVariableVectorBounds(const std::string &name) {
    int id = GetDecisionVariablesId(name);
    assert(id >= 0 && "Variable not found in program!\n");
    lbx_.middleRows(variables_idx_[id], variables_[id].sz()) =
        variables_[id].LowerBound();
    ubx_.middleRows(variables_idx_[id], variables_[id].sz()) =
        variables_[id].UpperBound();
}

void Program::UpdateDecisionVariableVectorBounds() {
    for (auto &v : variables_id_) {
        UpdateDecisionVariableVectorBounds(v.first);
    }
}

void Program::UpdateConstraintVectorBounds(const std::string &name) {
    int id = GetConstraintId(name);
    assert(id >= 0 && "Constraint not found in program!\n");
    lbg_.middleRows(constraints_idx_[id], constraints_[id].dim()) =
        constraints_[id].lb();
    ubg_.middleRows(constraints_idx_[id], constraints_[id].dim()) =
        constraints_[id].ub();
}

void Program::UpdateConstraintVectorBounds() {
    for (auto &c : constraints_id_) {
        UpdateConstraintVectorBounds(c.first);
    }
}

inline bool Program::IsParameter(const std::string &name) {
    // Indicate if it is present in the parameter map
    return parameters_id_.find(name) != parameters_id_.end();
}

int Program::GetParametersId(const std::string &name) {
    auto p = parameters_id_.find(name);
    if (p != parameters_id_.end()) {
        return p->second;
    } else {
        return -1;
    }
}

int Program::GetCostId(const std::string &name) {
    auto p = costs_id_.find(name);
    if (p != costs_id_.end()) {
        return p->second;
    } else {
        return -1;
    }
}

int Program::GetConstraintId(const std::string &name) {
    auto p = constraints_id_.find(name);
    if (p != constraints_id_.end()) {
        return p->second;
    } else {
        return -1;
    }
}

Variable &Program::GetParameters(const std::string &name) {
    int id = GetParametersId(name);
    assert(id >= 0 && "Parameter not found in program!\n");
    return parameters_[id];
}

Constraint &Program::GetConstraint(const std::string &name) {
    int id = GetConstraintId(name);
    assert(id >= 0 && "Constraint not found in program!\n");
    return constraints_[id];
}

Cost &Program::GetCost(const std::string &name) {
    int id = GetCostId(name);
    assert(id >= 0 && "Cost not found in program!\n");
    return costs_[id];
}

void Program::RemoveParameters(const std::string &name) {
    // Delete the entry and re-id other parameters
    int id = GetParametersId(name);
    assert(id >= 0 && "Parameter not found in program!\n");

    // Delete parameter
    parameters_id_.erase(name);
    parameters_.erase(std::next(parameters_.begin() + id));

    // Re-id remaining parameters
    for (int i = 0; i < parameters_.size(); ++i) {
        parameters_id_[parameters_[i].name()] = i;
    }
}

void Program::RemoveConstraint(const std::string &name) {
    // Delete the entry
    int id = GetConstraintId(name);
    assert(id >= 0 && "Constraint not found in program!\n");

    // Delete constraint
    constraints_id_.erase(name);
    constraints_.erase(std::next(constraints_.begin() + id));

    // Re-id remaining constraints
    for (int i = 0; i < constraints_.size(); ++i) {
        constraints_id_[constraints_[i].name()] = i;
    }
}

void Program::RemoveCost(const std::string &name) {
    int id = GetCostId(name);
    assert(id >= 0 && "Cost not found in program!\n");

    // Delete cost
    costs_id_.erase(name);
    costs_.erase(std::next(costs_.begin() + id));

    // Re-id remaining costs
    for (int i = 0; i < costs_.size(); ++i) {
        costs_id_[costs_[i].name()] = i;
    }
}

int Program::GetDecisionVariablesIndex(const std::string &name) {
    int id = GetDecisionVariablesId(name);
    if (id >= 0) {
        return variables_idx_[variables_id_[name]];
    } else {
        std::cout << "Variable " << name << " is not within this program!\n";
        return -1;
    }
}

int Program::GetConstraintIndex(const std::string &name) {
    int id = GetConstraintId(name);
    if (id >= 0) {
        return constraints_idx_[constraints_id_[name]];
    } else {
        std::cout << "Constraint " << name << " is not within this program!\n";
        return -1;
    }
}

void Program::AddParameters(const std::string &name, const int n, const int m) {
    damotion::common::Profiler("Program::AddParameters");
    if (!IsParameter(name)) {
        parameters_id_[name] = parameters_.size();
        parameters_.push_back(Variable(name, n, m));
    } else {
        std::cout << "Parameter with name " << name
                  << " already exist in program!\n";
    }
}

void Program::SetParameters(const std::string &name,
                            const Eigen::MatrixXd &val) {
    int id = GetParametersId(name);
    if (id >= 0) {
        parameters_[id].val() << val;
    }
}

void Program::SetFunctionInputData(utils::casadi::FunctionWrapper &f) {
    // Go through all function inputs and set both variable and parameter
    // locations
    for (int i = 0; i < f.f().n_in(); ++i) {
        std::string name = f.f().name_in(i);
        if (IsDecisionVariable(name)) {
            // Set variable input data as sub-vector of optimisation vector
            f.setInput(f.f().index_in(name),
                       x_.val().data() + variables_idx_[variables_id_[name]]);
        } else if (IsParameter(name)) {
            // Set parameter input data as vector in parameter map
            f.setInput(f.f().index_in(name),
                       parameters_[parameters_id_[name]].val());
        } else {
            // Throw warning that the input was not found
            std::cout << "Input " << name << " is not registered in program!\n";
        }
    }
}

void Program::ConstructDecisionVariableVector(
    const std::vector<std::string> &order) {
    // Vector of inputs
    casadi::SXVector x;

    int idx = 0;
    for (int i = 0; i < order.size(); ++i) {
        // Get id
        int id = GetDecisionVariablesId(order[i]);
        // Add to vector and set index
        x.push_back(variables_[id].sym());
        variables_idx_[id] = idx;
        idx += variables_[id].sz();
    }

    // Create optimisation vector
    x_ = Variable("x", idx);
    // Symbolic vector
    x_.sym() = casadi::SX::vertcat(x);

    // Create bounds for the variables
    lbx_.resize(x_.sz());
    ubx_.resize(x_.sz());

    lbx_.setConstant(-std::numeric_limits<double>::infinity());
    ubx_.setConstant(std::numeric_limits<double>::infinity());
}

void Program::ConstructConstraintVector() {
    // TODO - Make custom ordering possible
    nc_ = 0;
    for (Constraint &c : constraints_) {
        c.SetIndex(nc_);
        nc_ += c.dim();
    }

    // Create constraint bounds
    lbg_.resize(nc_);
    ubg_.resize(nc_);

    lbg_.setConstant(-std::numeric_limits<double>::infinity());
    ubg_.setConstant(std::numeric_limits<double>::infinity());

    // Set bounds
    for (Constraint &c : constraints_) {
        lbg_.middleRows(c.idx(), c.dim()) = c.lb();
        ubg_.middleRows(c.idx(), c.dim()) = c.ub();
    }
}

casadi::SXVector Program::GetSymbolicFunctionInput(
    const std::vector<std::string> &inames) {
    casadi::SXVector in = {};
    // Gather symbols for each input
    for (int i = 0; i < inames.size(); ++i) {
        std::string name = inames[i];
        if (IsDecisionVariable(name)) {
            in.push_back(GetDecisionVariables(name).sym());
        } else if (IsParameter(name)) {
            in.push_back(GetParameters(name).sym());
        } else {
            std::cout << "Input " << name
                      << " is not a listed variable or parameter!\n";
            // Return empty vector
            return {};
        }
    }

    // Return inputs
    return in;
}

std::string Program::GetSXName(const casadi::SX &x) {
    // Get first element of the SX matrix
    std::string name = x(0).name();
    // Remove trailing '_0' to get variable name
    if (name.substr(name.size() - 2) == "_0") {
        name = name.substr(0, name.size() - 2);
    }

    // Return the name of the vector
    return name;
}

casadi::StringVector Program::GetSXVectorNames(const casadi::SXVector &x) {
    casadi::StringVector inames = {};

    for (casadi::SX xi : x) {
        inames.push_back(GetSXName(xi));
    }

    return inames;
}

void Program::AddCost(const std::string &name, casadi::SX &cost,
                      casadi::SXVector &in) {
    int id = GetCostId(name);
    if (id >= 0) {
        std::cout << "Cost " << name << " already exists within the program!\n";
        return;
    }

    // Create cost and establish symbolic inputs and value
    Cost c(name);
    c.SetSymbolicObjective(cost);
    c.SetSymbolicInputs(in);

    // Add to map
    costs_id_[name] = costs_.size();
    costs_.push_back(c);
}

void Program::AddConstraint(const std::string &name, casadi::SX &constraint,
                            casadi::SXVector &in, const BoundsType &bounds) {
    int id = GetConstraintId(name);
    if (id >= 0) {
        std::cout << "Constraint " << name
                  << " already exists within the program!\n";
        return;
    }

    // Create cost and establish symbolic inputs and value
    Constraint c(name, constraint.size1());
    c.SetSymbolicConstraint(constraint);
    c.SetSymbolicInputs(in);
    c.SetBoundsType(bounds);

    // Add to map
    constraints_id_[name] = constraints_.size();
    constraints_.push_back(c);
}

void Program::SetUpCost(Cost &cost, const casadi::SXVector &x) {
    // With initialised program, create cost using given vector
    casadi::StringVector inames = GetSXVectorNames(cost.SymbolicInputs());

    casadi::SX J = cost.SymbolicObjective();

    // Create functions to compute the necessary gradients and hessians
    /* Objective */
    casadi::Function obj =
        casadi::Function(cost.name() + "_obj", cost.SymbolicInputs(), {J},
                         inames, {cost.name() + "_obj"});

    cost.SetObjectiveFunction(obj);

    /* Objective Gradient */
    // ! See if this is going to work
    casadi::SXVector g;
    casadi::StringVector g_names;
    for (const casadi::SX &xi : x) {
        // Compute gradients of cost with respect to variable set x
        g.push_back(gradient(J, xi));
        g_names.push_back(cost.name() + "_grad_" + GetSXName(xi));
    }

    casadi::Function grad = casadi::Function(
        cost.name() + "_grad", cost.SymbolicInputs(), g, inames, g_names);
    cost.SetGradientFunction(grad);

    casadi::SXVector H;
    casadi::StringVector H_names;

    // Compute Hessians for given partitioning
    for (int i = 0; i < x.size(); ++i) {
        for (int j = i; i < x.size(); ++j) {
            // TODO: Look at making only upper triangular for diagonal terms
            H.push_back(jacobian(gradient(J, x[i]).T(), x[j]));
            H_names.push_back(cost.name() + "_hes_" + GetSXName(x[i]) + "_" +
                              GetSXName(x[j]));
        }
    }

    /* Objective Hessian */
    casadi::Function hes = casadi::Function(
        cost.name() + "_hes", cost.SymbolicInputs(), H, inames, H_names);

    cost.SetHessianFunction(hes);

    // Register functions with variable and parameter data
    SetFunctionInputData(cost.ObjectiveFunction());
    SetFunctionInputData(cost.GradientFunction());
    SetFunctionInputData(cost.HessianFunction());
}

void Program::SetUpCosts() {
    for (Cost &c : costs_) {
        // SetUpCost(c); // TODO - Fix this
    }
    // ! Delete data for the costs to free up space
}

void Program::SetUpConstraint(Constraint &constraint) {
    // With initialised program, create constraint using given vector
    casadi::StringVector inames = GetSXVectorNames(constraint.SymbolicInputs());
    casadi::SX &x = DecisionVariableVector().sym();

    // Create functions to compute the necessary gradients and hessians

    /* Constraint */
    casadi::Function con(constraint.name() + "_con",
                         constraint.SymbolicInputs(),
                         {constraint.SymbolicConstraint()}, inames,
                         {constraint.name() + "_con"});

    /* Constraint Jacobian */
    casadi::Function jac(constraint.name() + "_jac",
                         constraint.SymbolicInputs(),
                         {jacobian(constraint.SymbolicConstraint(), x)}, inames,
                         {constraint.name() + "_jac"});

    /* Linearised constraint */
    // c(x) = A x + b = A(x - x0) + c(x0) = A x + (c(x0) - A x0)
    // casadi::SX A = jacobian(constraint.SymbolicConstraint(), x);
    // casadi::SX b = constraint.SymbolicConstraint() - casadi::SX::mtimes(A, x);

    // casadi::Function lin(constraint.name() + "_linearised",
    //                      {constraint.SymbolicInputs()}, {A, b}, inames,
    //                      {constraint.name() + "_A", constraint.name() + "_b"});

    constraint.SetConstraintFunction(con);
    constraint.SetJacobianFunction(jac);
    // constraint.SetLinearisedConstraintFunction(lin);

    // Set inputs for the functions
    SetFunctionInputData(constraint.ConstraintFunction());
    SetFunctionInputData(constraint.JacobianFunction());
    // SetFunctionInputData(constraint.LinearisedConstraintFunction());
}

void Program::SetUpConstraints() {
    for (Constraint &c : constraints_) {
        SetUpConstraint(c);
    }
    // ! Delete data for the costs to free up space
}

void Program::ListParameters() {
    std::cout << "----------------------\n";
    std::cout << "Parameter\tCurrent Value\n";
    std::cout << "----------------------\n";
    for (auto p : parameters_id_) {
        Variable &v = parameters_[p.second];
        for (int i = 0; i < v.rows(); i++) {
            for (int j = 0; j < v.cols(); j++) {
                std::cout << p.first << "_" << i << "_" << j << '\t'
                          << v.val()(i, j) << '\n';
            }
        }
    }
}

void Program::ListVariables() {
    std::cout << "----------------------\n";
    std::cout << "Variable\tCurrent Value\n";
    std::cout << "----------------------\n";
    for (auto p : variables_id_) {
        Variable &v = variables_[p.second];
        for (int i = 0; i < v.rows(); i++) {
            for (int j = 0; j < v.cols(); j++) {
                std::cout << p.first << "_" << i << "_" << j << '\t'
                          << v.val()(i, j) << '\n';
            }
        }
    }
}

void Program::ListConstraints() {
    std::cout << "----------------------\n";
    std::cout << "Constraint\tLower Bound\tUpper Bound\n";
    std::cout << "----------------------\n";
    for (auto p : constraints_id_) {
        for (int i = 0; i < constraints_[p.second].dim(); i++) {
            std::cout << p.first << "_" << i << '\t'
                      << constraints_[p.second].lb()(i) << '\t'
                      << constraints_[p.second].ub()(i) << '\n';
        }
    }
}

void Program::ListCosts() {
    std::cout << "----------------------\n";
    std::cout << "Cost\tWeigting\n";
    std::cout << "----------------------\n";
    for (auto p : costs_id_) {
        std::cout << p.first << '\t' << costs_[p.second].weighting() << '\n';
    }
}

void Program::PrintProgramSummary() {
    std::cout << "-----------------------\n";
    std::cout << "Program Name: " << name() << '\n';
    std::cout << "Number of Decision Variables: " << NumberOfDecisionVariables()
              << '\n';
    std::cout << "Variables\tSize\n";
    for (auto p : variables_id_) {
        Variable &v = variables_[p.second];
        std::cout << p.first << '\t' << v.sym().size() << '\n';
    }
    std::cout << "Number of Constraints: " << NumberOfConstraints() << '\n';
    std::cout << "Constraint\tSize\n";
    for (auto p : constraints_id_) {
        std::cout << p.first << "\t[" << constraints_[p.second].dim()
                  << ",1]\n";
    }
    std::cout << "Number of Parameters: "
              << "TBD" << '\n';
    std::cout << "Parameters\tSize\n";
    for (auto p : parameters_id_) {
        Variable &v = parameters_[p.second];
        std::cout << p.first << '\t' << v.sym().size() << '\n';
    }
    std::cout << "-----------------------\n";
}

}  // namespace optimisation
}  // namespace damotion