#include "solvers/program.h"

namespace damotion {
namespace optimisation {

void Program::AddDecisionVariables(const Eigen::Ref<sym::VariableMatrix> &var) {
    std::vector<sym::Variable> v;
    // Append to our map
    for (int i = 0; i < var.rows(); ++i) {
        for (int j = 0; j < var.cols(); ++j) {
            // Check if variable is already been included
            if (!IsDecisionVariable(var(i, j))) {
                // Add to decision variable id to keep track of variables
                decision_variable_id_.push_back(var(i, j).id());
                // Add to variable vector
                v.push_back(var(i, j));
                // Increase count of decision variables
                n_decision_variables_++;
            } else {
                // Variable already added to program!
                std::cout << var(i, j) << " is already added to program!\n";
            }
        }
    }

    // Add decision variables to problem
    decision_variables_.push_back(
        Eigen::Map<sym::VariableVector>(v.data(), v.size()));
}

bool Program::IsDecisionVariable(const sym::Variable &var) {
    // Check if variable is already added to the program
    return std::find(decision_variable_id_.begin(), decision_variable_id_.end(),
                     var.id()) != decision_variable_id_.end();
}

void Program::SetDecisionVariableVector() {
    // Set indices by order in the decision variable vector
    int start_idx = 0;
    for (int i = 0; i < decision_variables_.size(); ++i) {
        // Set start index of variables
        sym::Variable::Id id = decision_variables_[i].data()[0].id();
        decision_variable_start_idx_[id] = start_idx;
        // Increment the start index
        start_idx += decision_variables_[i].size();
    }
}

bool Program::SetDecisionVariableVector(
    const Eigen::Ref<sym::VariableVector> &var) {
    assert(var.size() == NumberOfDecisionVariables() && "Incorrect input!");

    // Set indices by order in the decision variable vector
    for (sym::VariableVector &v : decision_variables_) {
        // Find starting point of v within var
        int idx = 0;
        sym::Variable::Id id = v[0].id();
        for (int i = 0; i < var.size(); ++i) {
            if (var[idx].id() == id) break;
            idx++;
        }
        if (idx == var.size()) {
            std::cout << v
                      << " is not included within the provided decision "
                         "variable vector!\n";
            return false;
        } else {
            decision_variable_start_idx_[id] = idx;
        }
    }

    return true;
}

void Program::RemoveDecisionVariables(
    const Eigen::Ref<sym::VariableMatrix> &var) {
    // TODO - Implement
}

int Program::GetDecisionVariableStartIndex(const sym::VariableVector &v) {
    auto it = decision_variable_start_idx_.find(v[0].id());
    if (it != decision_variable_start_idx_.end()) {
        return it->second;
    } else {
        std::cout << v << " is not a variable within this program!\n";
        return -1;
    }
}

Eigen::Ref<const Eigen::MatrixXd> Program::AddParameters(
    const std::string &name, int n, int m) {
    // Check if parameter already exists
    auto it = parameters_.find(name);
    if (it != parameters_.end()) {
        std::cout << "Parameters with name " << name
                  << " already added to program!";
    } else {
        parameters_[name] = Eigen::MatrixXd::Zero(n, m);
        return parameters_[name];
    }
}

Eigen::Ref<const Eigen::MatrixXd> Program::GetParameters(
    const std::string &name) {
    auto it = parameters_.find(name);
    if (it == parameters_.end()) {
        std::cout << "Parameters with name " << name
                  << " is not in the program!";
    } else {
        return parameters_[name];
    }
}

void Program::SetParameters(const std::string &name,
                            Eigen::Ref<const Eigen::MatrixXd> val) {
    auto it = parameters_.find(name);
    if (it == parameters_.end()) {
        std::cout << "Parameters with name " << name
                  << " is not in the program!";
    } else {
        parameters_[name] = val;
    }
}

void Program::RemoveParameters(const std::string &name) {
    // Check if parameter exists
    auto it = parameters_.find(name);
    if (it == parameters_.end()) {
        std::cout << "Parameters with name " << name
                  << " is not in the program!";
    } else {
        parameters_.erase(name);
    }
}

Binding<Cost> Program::AddCost(const sym::Expression &cost,
                               const sym::VariableRefVector &x,
                               const sym::ParameterRefVector &p) {
    std::shared_ptr<Cost> c = std::make_shared<Cost>(cost, true, false);
    Binding<Cost> binding(c, x, p);
    costs_.push_back(binding);
    return costs_.back();
}

Binding<LinearConstraint> Program::AddLinearConstraint(
    const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
    const sym::VariableRefVector &x) {
    std::shared_ptr<LinearConstraint> p =
        std::make_shared<LinearConstraint>(A, b);
    linear_constraints_.push_back(Binding<LinearConstraint>(p, x));
    return linear_constraints_.back();
}

Binding<LinearConstraint> Program::AddLinearConstraint(
    const std::shared_ptr<LinearConstraint> &con,
    const sym::VariableRefVector &x, const sym::ParameterRefVector &p) {
    linear_constraints_.push_back(Binding<LinearConstraint>(con, x, p));
    return linear_constraints_.back();
}

Binding<Constraint> Program::AddConstraint(const std::shared_ptr<Constraint> &c,
                                           const sym::VariableRefVector &x,
                                           const sym::ParameterRefVector &p) {
    // Check bound variables and parameters exist
    for (const sym::VariableVector &v : x) {
        for (int i = 0; i < v.size(); i++) {
            if (!IsDecisionVariable(v[i])) {
                std::cout << v[i] << " is not included within this program!\n";
            }
        }
    }

    // Create a binding for the constraint
    constraints_.push_back(Binding<Constraint>(c, x, p));
    return constraints_.back();
}

Binding<Constraint> Program::AddGenericConstraint(
    const sym::Expression &c, const sym::VariableRefVector &x,
    const sym::ParameterRefVector &p) {
    std::shared_ptr<Constraint> con = std::make_shared<Constraint>(c);
    constraints_.push_back(Binding<Constraint>(con, x, p));
    return constraints_.back();
}

Binding<Constraint> Program::AddGenericConstraint(
    std::shared_ptr<Constraint> &c, const sym::VariableRefVector &x,
    const sym::ParameterRefVector &p) {
    constraints_.push_back(Binding<Constraint>(c, x, p));
    return constraints_.back();
}

void Program::ListParameters() {
    std::cout << "----------------------\n";
    std::cout << "Parameter\tCurrent Value\n";
    std::cout << "----------------------\n";
}

void Program::ListVariables() {
    std::cout << "----------------------\n";
    std::cout << "Variable\tCurrent Value\n";
    std::cout << "----------------------\n";
    for (sym::VariableVector &v : decision_variables_) {
        std::cout << v << '\n';
    }
}

void Program::ListConstraints() {
    std::cout << "----------------------\n";
    // Get all constraints
    std::vector<Binding<Constraint>> constraints = GetAllConstraints();
    for(Binding<Constraint> &b : constraints) {
        std::cout << b.Get().dim() << ",\t" << b.GetVariable(0) << std::endl;
    }
}

void Program::ListCosts() {
    std::cout << "----------------------\n";
    std::cout << "Cost\tWeigting\n";
    std::cout << "----------------------\n";
}

void Program::PrintProgramSummary() {
    std::cout << "-----------------------\n";
    std::cout << "Program Name: " << name() << '\n';
    std::cout << "Number of Decision Variables: " << NumberOfDecisionVariables()
              << '\n';
    std::cout << "Variables\tSize\n";

    std::cout << "Number of Constraints: " << NumberOfConstraints() << '\n';
    std::cout << "Constraint\tSize\n";

    std::cout << "Number of Parameters: "
              << "TBD" << '\n';
    std::cout << "Parameters\tSize\n";

    std::cout << "-----------------------\n";
}

}  // namespace optimisation
}  // namespace damotion