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
    const Eigen::Ref<const sym::VariableMatrix> &var) {
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
  for (const sym::Variable &xi : decision_variables_) {
    // Set index of each variable to idx
    decision_variable_idx_[xi.id()] = idx;
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

void ParameterManager::AddParameter(const sym::Parameter &p) {
  if (!IsParameter(p)) {
    // Add to parameter vector
    parameters_.push_back(p);
    parameter_idx_[p.id()] = n_parameters_;
    // Increase count of decision parameters
    n_parameters_++;
    parameter_vec_.conservativeResize(n_parameters_);
  } else {
    // Parameter already added to program!
    std::cout << p << " is already added to program!\n";
  }
}

void ParameterManager::AddParameters(const sym::ParameterVector &par) {
  // Append to our map
  for (int i = 0; i < par.size(); ++i) {
    AddParameter(par(i));
  }
}

void ParameterManager::AddParameters(const sym::ParameterMatrix &par) {
  for (int i = 0; i < par.rows(); ++i) {
    for (int j = 0; j < par.cols(); ++j) {
      AddParameter(par(i, j));
    }
  }
}

bool ParameterManager::IsParameter(const sym::Parameter &par) {
  // Check if parameter is already added to the program
  return std::find(parameters_.begin(), parameters_.end(), par) !=
         parameters_.end();
}

void ParameterManager::RemoveParameters(
    const Eigen::Ref<sym::ParameterMatrix> &var) {
  // TODO - Implement
}

int ParameterManager::GetParameterIndex(const sym::Parameter &p) {
  auto it = parameter_idx_.find(p.id());
  if (it != parameter_idx_.end()) {
    return it->second;
  } else {
    std::cout << p << " is not a parameter within this program!\n";
    return -1;
  }
}

bool ParameterManager::IsContinuousInParameterVector(
    const sym::ParameterVector &par) {
  VLOG(10) << "IsContinuousInParameterVector(), checking " << par;
  // Determine the index of the first element within par
  int idx = GetParameterIndex(par[0]);
  // Move through parameter vector and see if each entry follows one
  // after the other
  for (int i = 1; i < par.size(); ++i) {
    // If not, return false
    int idx_next = GetParameterIndex(par[i]);

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

void ParameterManager::ListParameters() {
  std::cout << "----------------------\n";
  std::cout << "parameter\n";
  std::cout << "----------------------\n";
  for (sym::Parameter &p : parameters_) {
    std::cout << p << '\n';
  }
}

}  // namespace optimisation
}  // namespace damotion
