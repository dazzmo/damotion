#include "solvers/program.h"

namespace damotion {
namespace optimisation {

bool Program::IsDecisionVariable(const std::string &name) {
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
    lbx_.middleRows(variables_[id].idx().i_start(), variables_[id].sz()) =
        variables_[id].LowerBound();
    ubx_.middleRows(variables_[id].idx().i_start(), variables_[id].sz()) =
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
    lbg_.middleRows(constraints_idx_[id].i_start(),
                    constraints_idx_[id].i_sz()) = constraints_[id].lb();
    ubg_.middleRows(constraints_idx_[id].i_start(),
                    constraints_idx_[id].i_sz()) = constraints_[id].ub();
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
        return variables_[variables_id_[name]].idx().i_start();
    } else {
        std::cout << "Variable " << name << " is not within this program!\n";
        return -1;
    }
}

int Program::GetConstraintIndex(const std::string &name) {
    int id = GetConstraintId(name);
    if (id >= 0) {
        return constraints_idx_[constraints_id_[name]].i_start();
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
            int id = variables_id_[name];
            f.setInput(f.f().index_in(name),
                       x_.val().data() + variables_[id].idx().i_start());
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
        variables_idx_[id] = BlockIndex(idx, variables_[id].sz());
        // Increase idx counter
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
        // Update row indexing for the constraint and its Jacobians
        int id = GetConstraintId(c.name());
        for (BlockIndex &idx : constraints_jac_idx_[id]) {
            idx.UpdateRowStartIndex(nc_);
        }
        nc_ += c.dim();
    }

    // Create constraint bounds
    lbg_.resize(nc_);
    ubg_.resize(nc_);

    lbg_.setConstant(-std::numeric_limits<double>::infinity());
    ubg_.setConstant(std::numeric_limits<double>::infinity());
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

void Program::AddCost(
    const std::string &name, casadi::SX &cost, casadi::SXVector &in,
    const casadi::SXVector &x, const casadi::StringVector xnames,
    const std::vector<std::pair<::casadi::SX, ::casadi::SX>> &xy,
    const std::vector<std::pair<std::string, std::string>> &xynames) {
    int id = GetCostId(name);
    if (id >= 0) {
        std::cout << "Cost " << name << " already exists within the program!\n";
        return;
    }

    casadi::SX &xv = DecisionVariableVector().sym();

    // Create cost and establish symbolic inputs and value
    Cost c(name);
    c.SetSymbolicObjective(cost);
    c.SetSymbolicInputs(in);

    casadi::Function obj, grad, hes;

    // Create cost function
    casadi::StringVector inames = GetSXVectorNames(in);
    obj = casadi::Function(name + "_obj", in, {cost}, inames, {name + "_obj"});

    c.SetObjectiveFunction(obj);

    // Add indexing data for gradient and hessian calculations
    costs_grad_idx_.push_back({});
    costs_hes_idx_.push_back({});

    if (x.size()) {
        // Compute gradient with respect to custom variable set x
        grad = utils::casadi::CreateGradientFunction(name, cost, in, inames, x,
                                                     xnames);
        // Add indexing data for the gradient blocks
        costs_grad_idx_[id].resize(xnames.size());
        for (int i = 0; i < xnames.size(); ++i) {
            costs_grad_idx_[id][i] =
                BlockIndex(GetDecisionVariablesIndex(xnames[i]),
                           GetDecisionVariables(xnames[i]).sz());
        }

    } else {
        // Compute gradient with respect to optimisation vector only
        grad = utils::casadi::CreateGradientFunction(name, cost, in, inames,
                                                     {xv}, {"x"});
        // Add indexing data for the gradient
        costs_grad_idx_[id] = {BlockIndex(0, DecisionVariableVector().sz())};
    }
    c.SetGradientFunction(grad);

    if (xy.size()) {
        // Compute hessian with respect to custom variable set x
        hes = utils::casadi::CreateHessianFunction(name, cost, in, inames, xy,
                                                   xynames);
        // Add indexing data for the hessian blocks
        // Add indexing data for the gradient blocks
        costs_hes_idx_[id].resize(xnames.size());
        for (int i = 0; i < xnames.size(); ++i) {
            costs_hes_idx_[id][i] =
                BlockIndex(GetDecisionVariablesIndex(xynames[i].first),
                           GetDecisionVariables(xynames[i].first).sz(),
                           GetDecisionVariablesIndex(xynames[i].second),
                           GetDecisionVariables(xynames[i].second).sz());
        }
    } else {
        // Compute hessian with respect to optimisation vector only
        hes = utils::casadi::CreateHessianFunction(name, cost, in, inames,
                                                   {{xv, xv}}, {{"x", "x"}});
        // Add indexing data for the hessian
        costs_grad_idx_[id] = {BlockIndex(0, DecisionVariableVector().sz(), 0,
                                          DecisionVariableVector().sz())};
    }
    c.SetHessianFunction(hes);

    // Set up variable inputs
    SetFunctionInputData(c.ObjectiveFunction());
    SetFunctionInputData(c.GradientFunction());
    SetFunctionInputData(c.HessianFunction());

    // Add cost to map
    costs_id_[name] = costs_.size();
    costs_.push_back(c);
}

void Program::AddConstraint(
    const std::string &name, casadi::SX &constraint, casadi::SXVector &in,
    const BoundsType &bounds, const casadi::SXVector &x,
    const casadi::StringVector xnames,
    const std::vector<std::pair<::casadi::SX, ::casadi::SX>> &xy,
    const std::vector<std::pair<std::string, std::string>> &xynames) {
    int id = GetConstraintId(name);
    if (id >= 0) {
        std::cout << "Constraint " << name
                  << " already exists within the program!\n";
        return;
    }

    casadi::SX &xv = DecisionVariableVector().sym();

    // Create cost and establish symbolic inputs and value
    Constraint c(name, constraint.size1());
    c.SetSymbolicConstraint(constraint);
    c.SetSymbolicInputs(in);
    c.SetBoundsType(bounds);

    // Add constraint id
    constraints_id_[name] = constraints_.size();
    // Add indexing data for gradient and hessian calculations
    constraints_jac_idx_.push_back({});
    constraints_hes_idx_.push_back({});

    casadi::Function con, jac, hes;

    // Create cost function
    casadi::StringVector inames = GetSXVectorNames(in);
    con = casadi::Function(name + "_obj", in, {constraint}, inames,
                           {name + "_con"});

    c.SetConstraintFunction(con);

    if (x.size()) {
        // Compute gradient with respect to custom variable set x
        jac = utils::casadi::CreateJacobianFunction(name, constraint, in,
                                                    inames, x, xnames);
        // Add indexing data for the jacobian blocks
        constraints_jac_idx_[id].resize(xnames.size());
        for (int i = 0; i < xnames.size(); ++i) {
            constraints_jac_idx_[id][i] =
                BlockIndex(0, c.dim(), GetDecisionVariablesIndex(xnames[i]),
                           GetDecisionVariables(xnames[i]).sz());
        }
    } else {
        // Compute jacobian with respect to optimisation vector only
        jac = utils::casadi::CreateJacobianFunction(name, constraint, in,
                                                    inames, {xv}, {"x"});
        // Add indexing data for the hessian
        constraints_jac_idx_[id] = {
            BlockIndex(0, c.dim(), 0, DecisionVariableVector().sz())};
    }
    c.SetJacobianFunction(jac);

    // TODO - Hessian calculations for constraints for NLP
    if (xy.size()) {
        // Compute hessian with respect to custom variable set x

    } else {
        // Compute hessian with respect to optimisation vector only
    }
    // c.SetHessianFunction(hes);

    // Set up variable inputs
    SetFunctionInputData(c.ConstraintFunction());
    SetFunctionInputData(c.JacobianFunction());
    // SetFunctionInputData(c.HessianFunction());

    constraints_.push_back(c);
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