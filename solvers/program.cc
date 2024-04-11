#include "solvers/program.h"

namespace damotion {
namespace optimisation {

void Program::AddDecisionVariable(const sym::Variable &var) {
    if (!IsDecisionVariable(var)) {
        // Add to variable vector
        decision_variables_.push_back(var);
        // Increase count of decision variables
        n_decision_variables_++;
    } else {
        // Variable already added to program!
        std::cout << var << " is already added to program!\n";
    }
}

void Program::AddDecisionVariables(const Eigen::Ref<sym::VariableMatrix> &var) {
    std::vector<sym::Variable> v;
    // Append to our map
    for (int i = 0; i < var.rows(); ++i) {
        for (int j = 0; j < var.cols(); ++j) {
            AddDecisionVariable(var(i, j));
        }
    }
}

bool Program::IsDecisionVariable(const sym::Variable &var) {
    // Check if variable is already added to the program
    return std::find(decision_variables_.begin(), decision_variables_.end(),
                     var) != decision_variables_.end();
}

void Program::SetDecisionVariableVector() {
    // Set indices by order in the decision variable vector
    int idx = 0;
    for (int i = 0; i < decision_variables_.size(); ++i) {
        // Set start index of variables
        sym::Variable::Id id = decision_variables_[i].id();
        decision_variable_idx_[id] = idx;
        // Increment the start index
        idx++;
    }

    // Create default variable bounds
    double inf = std::numeric_limits<double>::infinity();
    lbx_ = -inf * Eigen::VectorXd::Ones(NumberOfDecisionVariables());
    ubx_ = inf * Eigen::VectorXd::Ones(NumberOfDecisionVariables());
}

bool Program::SetDecisionVariableVector(
    const Eigen::Ref<sym::VariableVector> &var) {
    assert(var.size() == NumberOfDecisionVariables() && "Incorrect input!");

    // Set indices by order in the decision variable vector
    for (sym::Variable &v : decision_variables_) {
        // Find starting point of v within var
        int idx = 0;
        sym::Variable::Id id = v.id();
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
            decision_variable_idx_[id] = idx;
        }
    }

    // Create default variable bounds
    double inf = std::numeric_limits<double>::infinity();
    lbx_ = -inf * Eigen::VectorXd::Ones(NumberOfDecisionVariables());
    ubx_ = inf * Eigen::VectorXd::Ones(NumberOfDecisionVariables());

    return true;
}

void Program::RemoveDecisionVariables(
    const Eigen::Ref<sym::VariableMatrix> &var) {
    // TODO - Implement
}

int Program::GetDecisionVariableIndex(const sym::Variable &v) {
    auto it = decision_variable_idx_.find(v.id());
    if (it != decision_variable_idx_.end()) {
        return it->second;
    } else {
        std::cout << v << " is not a variable within this program!\n";
        return -1;
    }
}

Eigen::Ref<Eigen::MatrixXd> Program::AddParameters(
    const std::string &name, int n, int m) {
    // Check if parameter already exists
    auto it = parameters_.find(name);
    if (it != parameters_.end()) {
        throw std::runtime_error("Parameters with name " + name +
                                 " already in the program!");
    } else {
        parameters_[name] = Eigen::MatrixXd::Zero(n, m);
        return parameters_[name];
    }
}

Eigen::Ref<Eigen::MatrixXd> Program::GetParameters(
    const std::string &name) {
    auto it = parameters_.find(name);
    if (it == parameters_.end()) {
        throw std::runtime_error("Parameters with name " + name +
                                 " is not in the program!");
    } else {
        return parameters_[name];
    }
}

void Program::SetParameters(const std::string &name,
                            Eigen::Ref<const Eigen::MatrixXd> val) {
    auto it = parameters_.find(name);
    if (it == parameters_.end()) {
        throw std::runtime_error("Parameters with name " + name +
                                 " is not in the program!");
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

Binding<Cost> Program::AddCost(const std::shared_ptr<Cost> &cost,
                               const sym::VariableRefVector &x,
                               const sym::ParameterRefVector &p) {
    Binding<Cost> binding(cost, x, p);
    costs_.push_back(binding);
    return costs_.back();
}

Binding<LinearCost> Program::AddLinearCost(
    const std::shared_ptr<LinearCost> &cost, const sym::VariableRefVector &x,
    const sym::ParameterRefVector &p) {
    linear_costs_.push_back(Binding<LinearCost>(cost, x, p));
    return linear_costs_.back();
}

Binding<QuadraticCost> Program::AddQuadraticCost(
    const std::shared_ptr<QuadraticCost> &cost, const sym::VariableRefVector &x,
    const sym::ParameterRefVector &p) {
    quadratic_costs_.push_back(Binding<QuadraticCost>(cost, x, p));
    return quadratic_costs_.back();
}

Binding<LinearConstraint> Program::AddLinearConstraint(
    const std::shared_ptr<LinearConstraint> &con,
    const sym::VariableRefVector &x, const sym::ParameterRefVector &p) {
    linear_constraints_.push_back(Binding<LinearConstraint>(con, x, p));
    n_constraints_ += con->Dimension();
    return linear_constraints_.back();
}

Binding<BoundingBoxConstraint> Program::AddBoundingBoxConstraint(
    const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
    const sym::VariableVector &x) {
    std::shared_ptr<BoundingBoxConstraint> con =
        std::make_shared<BoundingBoxConstraint>("", lb, ub);
    bounding_box_constraints_.push_back(
        Binding<BoundingBoxConstraint>(con, {x}));
    return bounding_box_constraints_.back();
}

Binding<BoundingBoxConstraint> Program::AddBoundingBoxConstraint(
    const double &lb, const double &ub, const sym::VariableVector &x) {
    Eigen::VectorXd lbv(x.size()), ubv(x.size());
    lbv.setConstant(lb);
    ubv.setConstant(ub);
    return AddBoundingBoxConstraint(lbv, ubv, x);
}

Binding<Constraint> Program::AddConstraint(
    const std::shared_ptr<Constraint> &con, const sym::VariableRefVector &x,
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
    constraints_.push_back(Binding<Constraint>(con, x, p));
    n_constraints_ += con->Dimension();
    return constraints_.back();
}

Binding<Constraint> Program::AddGenericConstraint(
    std::shared_ptr<Constraint> &con, const sym::VariableRefVector &x,
    const sym::ParameterRefVector &p) {
    constraints_.push_back(Binding<Constraint>(con, x, p));
    n_constraints_ += con->Dimension();
    return constraints_.back();
}

void Program::ListParameters() {
    std::cout << "----------------------\n";
    std::cout << "Parameter\tCurrent Value\n";
    std::cout << "----------------------\n";
    for (const auto &p : parameters_) {
        std::cout << p.first << '\t' << p.second << '\n';
    }
}

void Program::ListDecisionVariables() {
    std::cout << "----------------------\n";
    std::cout << "Variable\tCurrent Value\n";
    std::cout << "----------------------\n";
    for (sym::Variable &v : decision_variables_) {
        std::cout << v << '\n';
    }
}

void Program::ListConstraints() {
    std::cout << "----------------------\n";
    std::cout << "Constraint\tSize\tUpper Bound\tLower Bound\n";
    std::cout << "----------------------\n";
    // Get all constraints
    std::vector<Binding<Constraint>> constraints = GetAllConstraints();
    for (Binding<Constraint> &b : constraints) {
        std::cout << b.Get().name() << "\t[" << b.Get().Dimension() << ",1]\n";
        for (int i = 0; i < b.Get().Dimension(); ++i) {
            std::cout << b.Get().name() << "_" + std::to_string(i) << "\t\t"
                      << b.Get().UpperBound()[i] << "\t"
                      << b.Get().LowerBound()[i] << "\n";
        }
    }
}

void Program::ListCosts() {
    std::cout << "----------------------\n";
    std::cout << "Cost\tWeighting\n";
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