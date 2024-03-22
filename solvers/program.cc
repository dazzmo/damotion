#include "solvers/program.h"

namespace damotion {
namespace optimisation {

void Program::VariableNotFoundError(const std::string &name) {
    std::cout << "Variable with name " << name
              << " is not included within this program!\n";
}

void Program::ParameterNotFoundError(const std::string &name) {
    std::cout << "Parameter with name " << name
              << " is not included within this program!\n";
}

void VariableAlreadyExistsError(const std::string &name) {
    std::cout << "Variables with name " << name << " already added!\n";
}
void ParameterAlreadyExistsError(const std::string &name) {
    std::cout << "Parameter with name " << name << " already added!\n";
}

inline bool Program::IsVariable(const std::string &name) {
    // Indicate if it is present in the variable map
    return variables_.find(name) != variables_.end();
}

inline bool Program::IsParameter(const std::string &name) {
    // Indicate if it is present in the parameter map
    return parameter_map_.find(name) != parameter_map_.end();
}

void Program::AddVariables(const std::string &name, const int sz) {
    if (!IsVariable(name)) {
        variables_[name] = casadi::SX::sym(name, sz);
        variable_map_[name] = Eigen::VectorXd::Zero(sz);
    } else {
        VariableAlreadyExistsError(name);
    }
}

casadi::SX &Program::GetVariables(const std::string &name) {
    if (!IsVariable(name)) {
        VariableNotFoundError(name);
    } else {
        return variables_[name];
    }
}

int Program::GetVariableIndex(const std::string &name) {
    if (!IsVariable(name)) {
        VariableNotFoundError(name);
        return -1;
    } else {
        return variable_idx_[name];
    }
}

Eigen::Ref<Eigen::VectorXd> Program::GetVariableUpperBounds(
    const std::string &name) {
    if (IsVariable(name)) {
        return ubx_.middleRows(variable_idx_[name], variable_map_[name].size());
    } else {
        throw std::runtime_error(name + "is not a variable in this program");
    }
}

Eigen::Ref<Eigen::VectorXd> Program::GetVariableLowerBounds(
    const std::string &name) {
    if (IsVariable(name)) {
        return lbx_.middleRows(variable_idx_[name], variable_map_[name].size());
    } else {
        throw std::runtime_error(name + "is not a variable in this program");
    }
}

void Program::SetDecisionVariablesFromVector(const Eigen::VectorXd &x) {
    damotion::common::Profiler profiler("Program::UpdateDecisionVariables");
    // Use current mapping to compute variables
    for (auto &xi : variables_) {
        variable_map_[xi.first] = x.middleRows(variable_idx_[xi.first],
                                               variable_map_[xi.first].size());
    }
}

void Program::ResizeDecisionVariables(const std::string &name, const int &sz) {
    auto p = variables_.find(name);
    // If it exists, update parameter
    if (p != variables_.end()) {
        // Create new symbolic expression and data vector
        variables_[name] = casadi::SX::sym(name, sz);
        variable_map_[name] = Eigen::VectorXd::Zero(sz);
    } else {
        VariableNotFoundError(name);
    }
}

casadi::SX &Program::GetParameters(const std::string &name) {
    if (!IsParameter(name)) {
        ParameterNotFoundError(name);
    } else {
        return parameters_[name];
    }
}

void Program::SetParameter(const std::string &name,
                           const Eigen::VectorXd &val) {
    // Look up parameters in parameter map and set values
    auto p = parameter_map_.find(name);
    // If it exists, update parameter
    if (p != parameter_map_.end()) {
        p->second = val;
    } else {
        ParameterNotFoundError(name);
    }
}

void Program::AddParameters(const std::string &name, int sz) {
    damotion::common::Profiler("Program::AddParameters");
    // If doesn't exist, add parameter
    if (!IsParameter(name)) {
        parameters_[name] = casadi::SX::sym(name, sz);
        parameter_map_[name] = Eigen::VectorXd::Zero(sz);
    } else {
        ParameterAlreadyExistsError(name);
    }
}

void Program::SetFunctionInputData(utils::casadi::FunctionWrapper &f) {
    // Go through all function inputs and set both variable and parameter
    // locations
    for (int i = 0; i < f.f().n_in(); ++i) {
        std::string name = f.f().name_in(i);
        if (IsVariable(name)) {
            f.setInput(f.f().index_in(name), variable_map_[name]);
        } else if (IsParameter(name)) {
            f.setInput(f.f().index_in(name), parameter_map_[name]);
        } else {
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
        x.push_back(GetVariables(order[i]));
        variable_idx_[order[i]] = idx;
        idx += GetVariables(order[i]).size1();
    }

    // Create optimisation vector
    x_ = casadi::SX::vertcat(x);

    // Set number of decision variables
    nx_ = x_.size1();

    // Create bounds for the variables
    lbx_.resize(nx_);
    ubx_.resize(nx_);

    lbx_.setConstant(-std::numeric_limits<double>::infinity());
    ubx_.setConstant(std::numeric_limits<double>::infinity());
}

void Program::ConstructConstraintVector() {
    // TODO - Make custom ordering possible
    nc_ = 0;
    for (auto &p : constraints_) {
        Constraint &c = p.second;
        c.SetIndex(nc_);
        nc_ += c.dim();
    }

    // Create constraint bounds
    lbg_.resize(nc_);
    ubg_.resize(nc_);

    lbg_.setConstant(-std::numeric_limits<double>::infinity());
    ubg_.setConstant(std::numeric_limits<double>::infinity());

    // Set bounds
    for (auto &p : constraints_) {
        Constraint &c = p.second;
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
        if (IsVariable(name)) {
            in.push_back(GetVariables(name));
        } else if (IsParameter(name)) {
            in.push_back(GetParameters(name));
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

casadi::StringVector Program::GetSXVectorNames(const casadi::SXVector &x) {
    casadi::StringVector inames = {};

    for (casadi::SX xi : x) {
        // Get the name of the symbol
        std::string name = xi(0).name();

        // Remove trailing '_0' to get variable name
        if (name.substr(name.size() - 2) == "_0") {
            name = name.substr(0, name.size() - 2);
        }

        inames.push_back(name);
    }

    return inames;
}

void Program::AddCost(const std::string &name, casadi::SX &cost,
                      casadi::SXVector &in) {
    // Make sure it doesn't exist already
    if (costs_.find(name) != costs_.end()) {
        std::cout << "Cost " << name << " already exists within the program!\n";
    }

    // Create cost and establish symbolic inputs and value
    Cost c(name);
    c.SetSymbolicObjective(cost);
    c.SetSymbolicInputs(in);

    // Add to map
    costs_[name] = c;
}

void Program::AddConstraint(const std::string &name, casadi::SX &constraint,
                            casadi::SXVector &in, const BoundsType &bounds) {
    // Make sure it doesn't exist already
    if (constraints_.find(name) != constraints_.end()) {
        std::cout << "Constraint " << name
                  << " already exists within the program!\n";
    }

    // Create cost and establish symbolic inputs and value
    Constraint c(name, constraint.size1());
    c.SetSymbolicConstraint(constraint);
    c.SetSymbolicInputs(in);
    c.SetBoundsType(bounds);

    // Add to map
    constraints_[name] = c;
}

void Program::SetUpCost(Cost &cost) {
    // With initialised program, create cost using given vector
    casadi::StringVector inames = GetSXVectorNames(cost.SymbolicInputs());
    casadi::SX &x = DecisionVariableVector();

    // Create functions to compute the necessary gradients and hessians
    /* Objective */
    casadi::Function obj = casadi::Function(
        cost.name() + "_obj", cost.SymbolicInputs(), {cost.SymbolicObjective()},
        inames, {cost.name() + "_obj"});

    cost.SetObjectiveFunction(obj);

    /* Objective Gradient */
    casadi::Function grad =
        casadi::Function(cost.name() + "_grad", cost.SymbolicInputs(),
                         {gradient(cost.SymbolicObjective(), x)}, inames,
                         {cost.name() + "_grad"});
    cost.SetGradientFunction(grad);

    /* Objective Hessian */
    casadi::Function hes = casadi::Function(
        cost.name() + "_hes", cost.SymbolicInputs(),
        {hessian(cost.SymbolicObjective(), x)}, inames, {cost.name() + "_hes"});

    cost.SetHessianFunction(hes);

    // Register functions with variable and parameter data
    SetFunctionInputData(cost.ObjectiveFunction());
    SetFunctionInputData(cost.GradientFunction());
    SetFunctionInputData(cost.HessianFunction());
}

void Program::SetUpCosts() {
    for (auto &c : costs_) {
        SetUpCost(c.second);
    }
    // ! Delete data for the costs to free up space
}

void Program::SetUpConstraint(Constraint &constraint) {
    // With initialised program, create constraint using given vector
    casadi::StringVector inames = GetSXVectorNames(constraint.SymbolicInputs());
    casadi::SX &x = DecisionVariableVector();

    // Create functions to compute the necessary gradients and hessians

    /* Constraint */
    casadi::Function con(constraint.name() + "_con",
                         constraint.SymbolicInputs(),
                         {constraint.SymbolicConstraint()}, inames,
                         {constraint.name() + "_con"});

    /* Objective Gradient */
    casadi::Function jac(constraint.name() + "_jac",
                         constraint.SymbolicInputs(),
                         {jacobian(constraint.SymbolicConstraint(), x)}, inames,
                         {constraint.name() + "_jac"});

    constraint.SetConstraintFunction(con);
    constraint.SetJacobianFunction(jac);

    // Set inputs for the functions
    SetFunctionInputData(constraint.ConstraintFunction());
    SetFunctionInputData(constraint.JacobianFunction());
}

void Program::ListParameters() {
    std::cout << "----------------------\n";
    std::cout << "Parameter\tCurrent Value\n";
    std::cout << "----------------------\n";
    for (auto p : parameter_map_) {
        for (int i = 0; i < p.second.size(); i++) {
            std::cout << p.first << "_" << i << '\t' << p.second(i) << '\n';
        }
    }
}

void Program::ListVariables() {
    std::cout << "----------------------\n";
    std::cout << "Variable\tSize\n";
    std::cout << "----------------------\n";
    for (auto v : variables_) {
        std::cout << v.first << '\t' << v.second.size() << '\n';
    }
}

void Program::ListConstraints() {
    std::cout << "----------------------\n";
    std::cout << "Constraint\tLower Bound\tUpper Bound\n";
    std::cout << "----------------------\n";
    for (auto c : constraints_) {
        for (int i = 0; i < c.second.dim(); i++) {
            std::cout << c.first << "_" << i << '\t' << c.second.lb()(i) << '\t'
                      << c.second.ub()(i) << '\n';
        }
    }
}

void Program::ListCosts() {
    std::cout << "----------------------\n";
    std::cout << "Cost\tWeigting\n";
    std::cout << "----------------------\n";
    for (auto c : costs_) {
        std::cout << c.first << '\t' << c.second.weighting() << '\n';
    }
}

void Program::PrintProgramSummary() {
    std::cout << "-----------------------\n";
    std::cout << "Program Name: " << name() << '\n';
    std::cout << "Number of Decision Variables: " << NumberOfDecisionVariables()
              << '\n';
    std::cout << "Variables\tSize\n";
    for (auto v : variables_) {
        std::cout << v.first << '\t' << v.second.size() << '\n';
    }
    std::cout << "Number of Constraints: " << NumberOfConstraints() << '\n';
    std::cout << "Constraint\tSize\n";
    for (auto c : constraints_) {
        std::cout << c.first << "\t[" << c.second.dim() << ",1]\n";
    }
    // ! Fix the number of parameters
    std::cout << "Number of Parameters: "
              << "TBD" << '\n';
    std::cout << "Parameters\tSize\n";
    for (auto p : parameters_) {
        std::cout << p.first << '\t' << p.second.size() << '\n';
    }
    std::cout << "-----------------------\n";
}

}  // namespace optimisation
}  // namespace damotion