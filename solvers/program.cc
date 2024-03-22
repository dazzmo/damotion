#include "solvers/program.h"

namespace damotion {

namespace solvers {

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
}

void Program::AddConstraint(const std::string &name, casadi::SX &constraint,
                            casadi::SXVector &in) {}

void Program::RegisterCost(Cost &cost) {
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
    SetFunctionData(cost.ObjectiveFunction());
    SetFunctionData(cost.GradientFunction());
    SetFunctionData(cost.HessianFunction());

    // Add to the cost map
    if (costs_.find(cost.name()) != costs_.end()) {
        std::cerr << "Cost with name " << cost.name()
                  << " already registered in program!\n";
    } else {
        costs_[cost.name()] = cost;
    }
}

void Program::RegisterCosts() {
    for (auto &c : costs_) {
        RegisterCost(c.second);
    }
    // ! Delete data for the costs to free up space
}

void Program::RegisterConstraint(const Constraint &constraint) {
    // With initialised program, create constraint using given vector
    std::string name = data.name;
    casadi::StringVector inames = GetSXVectorNames(data.inputs);
    casadi::SX &x = DecisionVariableVector();

    // Create functions to compute the necessary gradients and hessians

    /* Constraint */
    casadi::Function con(name + "_con", data.inputs, {data.con}, inames,
                         {name + "_con"});

    /* Objective Gradient */
    casadi::Function jac(name + "_jac", data.inputs, {jacobian(data.con, x)},
                         inames, {name + "_jac"});

    Constraint constraint(data.name, data.con.size1());

    constraint.SetConstraintFunction(con);
    constraint.SetJacobianFunction(jac);

    // Set upper and lower bounds
    constraint.lb() = data.lb;
    constraint.ub() = data.ub;

    // Set inputs for the functions
    SetFunctionData(constraint.ConstraintFunction());
    SetFunctionData(constraint.JacobianFunction());

    // Add to the cost map
    if (constraints_.find(name) != constraints_.end()) {
        std::cerr << "Constraint with name " << name
                  << " already registered in program!\n";
    } else {
        constraints_[name] = constraint;
    }
}

void Program::RegisterConstraints() {
    for (const ConstraintData &c : constraint_data_) {
        RegisterConstraint(c);
    }
    // ! Delete data for the costs to free up space
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

}  // namespace solvers
}  // namespace damotion