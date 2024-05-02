#include "damotion/optimisation/program.h"

namespace damotion {
namespace optimisation {

void DecisionVariableManager::AddDecisionVariable(const sym::Variable &var) {
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

void DecisionVariableManager::AddDecisionVariables(
    const Eigen::Ref<sym::VariableMatrix> &var) {
  std::vector<sym::Variable> v;
  // Append to our map
  for (int i = 0; i < var.rows(); ++i) {
    for (int j = 0; j < var.cols(); ++j) {
      AddDecisionVariable(var(i, j));
    }
  }
}

bool DecisionVariableManager::IsDecisionVariable(const sym::Variable &var) {
  // Check if variable is already added to the program
  return std::find(decision_variables_.begin(), decision_variables_.end(),
                   var) != decision_variables_.end();
}

void DecisionVariableManager::SetDecisionVariableVector() {
  // Set indices by order in the decision variable vector
  int idx = 0;
  for (int i = 0; i < decision_variables_.size(); ++i) {
    // Set start index of variables
    sym::Variable::Id id = decision_variables_[i].id();
    decision_variable_idx_[id] = idx;
    // Increment the start index
    idx++;
  }
}

bool DecisionVariableManager::SetDecisionVariableVector(
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

  return true;
}

void DecisionVariableManager::RemoveDecisionVariables(
    const Eigen::Ref<sym::VariableMatrix> &var) {
  // TODO - Implement
}

int DecisionVariableManager::GetDecisionVariableIndex(const sym::Variable &v) {
  auto it = decision_variable_idx_.find(v.id());
  if (it != decision_variable_idx_.end()) {
    return it->second;
  } else {
    std::cout << v << " is not a variable within this program!\n";
    return -1;
  }
}

bool DecisionVariableManager::IsContinuousInDecisionVariableVector(
    const sym::VariableVector &var) {
  VLOG(10) << "IsContinuousInDecisionVariableVector(), checking " << var;
  // Determine the index of the first element within var
  int idx = GetDecisionVariableIndex(var[0]);
  // Move through optimisation vector and see if each entry follows one
  // after the other
  for (int i = 1; i < var.size(); ++i) {
    // If not, return false
    int idx_next = GetDecisionVariableIndex(var[i]);

    if (idx_next - idx != 1) {
      VLOG(10) << "false";
      return false;
    }
    idx = idx_next;
  }
  // Return true if all together in the vector
  VLOG(10) << "true";
  return true;
}

void DecisionVariableManager::ListDecisionVariables() {
  std::cout << "----------------------\n";
  std::cout << "Variable\n";
  std::cout << "----------------------\n";
  for (sym::Variable &v : decision_variables_) {
    std::cout << v << '\n';
  }
}

bool ParameterManager::IsParameter(const sym::Parameter &par) {
  // Check if variable is already added to the program
  return parameter_idx_.find(par.id()) != parameter_idx_.end();
}

Eigen::Ref<Eigen::MatrixXd> ParameterManager::AddParameter(
    const sym::Parameter &p) {
  // Check if parameter already exists
  if (IsParameter(p)) {
    throw std::runtime_error("Parameter " + p.name() +
                             " already in the program!");
  } else {
    // Add parameter values
    parameter_idx_[p.id()] = parameters_.size();
    parameters_.push_back(p);
    parameter_vals_.push_back(Eigen::MatrixXd::Zero(p.rows(), p.cols()));
    n_parameters_ += p.rows() * p.cols();
    return parameter_vals_.back();
  }
}

Eigen::Ref<const Eigen::MatrixXd> ParameterManager::GetParameterValues(
    const sym::Parameter &p) {
  auto it = parameter_idx_.find(p.id());
  if (it == parameter_idx_.end()) {
    throw std::runtime_error("Parameter " + p.name() +
                             " is not in the program!");
  } else {
    return parameter_vals_[it->second];
  }
}

void ParameterManager::SetParameterValues(
    const sym::Parameter &p, Eigen::Ref<const Eigen::MatrixXd> val) {
  auto it = parameter_idx_.find(p.id());
  if (it == parameter_idx_.end()) {
    throw std::runtime_error("Parameter " + p.name() +
                             " is not in the program!");
  } else {
    assert(val.rows() == p.rows() && val.cols() == p.cols() &&
           "Parameter dimension mismatch!");
    parameter_vals_[it->second] = val;
  }
}

void ParameterManager::RemoveParameters(const sym::Parameter &p) {
  // Check if parameter exists
  auto it = parameter_idx_.find(p.id());
  if (it == parameter_idx_.end()) {
    std::cout << "Parameter " << p.name() << " is not in the program!";
  } else {
    parameters_.erase(parameters_.begin() + it->second);
    parameter_vals_.erase(parameter_vals_.begin() + it->second);
    parameter_idx_.erase(p.id());
    n_parameters_ -= p.rows() * p.cols();
  }
}

void ParameterManager::ListParameters() {
  std::cout << "----------------------\n";
  std::cout << "Parameter\tCurrent Value\n";
  std::cout << "----------------------\n";
  for (const auto &p : parameters_) {
    std::cout << p.name() << '\t' << parameter_vals_[parameter_idx_[p.id()]]
              << '\n';
  }
}
}  // namespace optimisation
}  // namespace damotion
