#include "damotion/symbolic/variable.h"
namespace damotion {
namespace symbolic {

// Variable matrix
Matrix createMatrix(const std::string &name, const int m, const int n) {
  Matrix mat(m, n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      mat(i, j) =
          Variable(name + '_' + std::to_string(i) + '_' + std::to_string(j));
    }
  }
  return mat;
}

// Variable vector
Vector createVector(const std::string &name, const int n) {
  Vector vec(n);
  for (int i = 0; i < n; ++i) {
    vec[i] = Variable(name + '_' + std::to_string(i));
  }
  return vec;
}

// Create vector of decision variables
Vector concatenateVariables(const VectorRefList &vars) {
  Vector vec;
  for (const auto &var : vars) {
    vec.conservativeResize(vec.size() + var.size());
    vec.bottomRows(var.size()) = var;
  }
  return vec;
}

Vector concatenateVariables(const MatrixRefList &vars) {
  Vector vec;
  for (const auto &var : vars) {
    // Flatten variable matrices where applicable
    Vector tmp = var.reshape(var.rows() * var.cols(), 1);
    vec.conservativeResize(vec.size() + tmp.size());
    vec.bottomRows(tmp.size()) = tmp;
  }
  return vec;
}

// Operator overloading

std::ostream &operator<<(std::ostream &os, damotion::symbolic::Variable var) {
  return os << var.name();
}

std::ostream &operator<<(std::ostream &os, damotion::symbolic::Vector vector) {
  std::ostringstream oss;
  for (int i = 0; i < vector.size(); i++) {
    oss << vector[i] << '\n';
  }
  return os << oss.str();
}

std::ostream &operator<<(std::ostream &os, damotion::symbolic::Matrix mat) {
  std::ostringstream oss;
  oss << "{\n";
  for (int i = 0; i < mat.rows(); i++) {
    oss << "  ";
    for (int j = 0; j < mat.cols(); j++) {
      oss << mat(i, j) << '\t';
    }
    oss << '\n';
  }
  oss << "}\n";
  return os << oss.str();
}

void VariableManager::addVariable(const sym::Variable &var) {
  if (!IsVariable(var)) {
    // Add to variable vector
    decision_variable_vec_idx_[var.id()] = decision_variables_.size();
    decision_variables_.push_back(var);
    decision_variables_data_.push_back(VariableData());
    // Increase count of decision variables
    n_decision_variables_++;
    xbl_.conservativeResize(n_decision_variables_);
    xbu_.conservativeResize(n_decision_variables_);
    x0_.conservativeResize(n_decision_variables_);
  } else {
    // Variable already added to program!
    std::cout << var << " is already added to program!\n";
  }
}

void VariableManager::addVariables(const MatrixRef &var) {
  // Append to our map
  for (int i = 0; i < var.rows(); ++i) {
    for (int j = 0; j < var.cols(); ++j) {
      addVariable(var(i, j));
    }
  }
}

bool VariableManager::isVariable(const sym::Variable &var) {
  // Check if variable is already added to the program
  return std::find(decision_variables_.begin(), decision_variables_.end(),
                   var) != decision_variables_.end();
}

void VariableManager::setVector() {
  // Set indices by order in the decision variable vector
  int idx = 0;
  for (const sym::Variable &xi : decision_variables_) {
    // Set index of each variable to idx
    decision_variable_idx_[xi.id()] = idx;
    idx++;
  }
}

bool VariableManager::setVector(const Eigen::Ref<sym::Vector> &var) {
  assert(var.size() == numberOfVariables() && "Incorrect input!");

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

void VariableManager::removeVariables(const MatrixRef &var) {
  // TODO - Implement
}

int VariableManager::getVariableIndex(const sym::Variable &v) {
  auto it = decision_variable_idx_.find(v.id());
  if (it != decision_variable_idx_.end()) {
    return it->second;
  } else {
    std::cout << v << " is not a variable within this program!\n";
    return -1;
  }
}

std::vector<int> VariableManager::getVariableIndices(const sym::Vector &v) {
  std::vector<int> indices;
  indices.reserve(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    int idx = getVariableIndex(v[i]);
    if (idx >= 0) {
      indices.push_back(idx);
    }
  }
  // Return vector of indices
  return indices;
}

bool VariableManager::isContinuousInVector(const sym::Vector &var) {
  VLOG(10) << "IsContinuousInVector(), checking " << var;
  // Determine the index of the first element within var
  int idx = getVariableIndex(var[0]);
  // Move through optimisation vector and see if each entry follows one
  // after the other
  for (int i = 1; i < var.size(); ++i) {
    // If not, return false
    int idx_next = getVariableIndex(var[i]);

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

void VariableManager::setVariableBounds(const sym::Variable &v,
                                        const double &bl, const double &bu) {
  auto it = decision_variable_vec_idx_.find(v.id());
  if (it != decision_variable_vec_idx_.end()) {
  } else {
    std::cout << v << " is not a variable within this program!\n";
    return;
  }
  VariableData &data = decision_variables_data_[it->second];
  data.bl = bl;
  data.bu = bu;
  data.bounds_updated = true;
}

void VariableManager::setVariableBounds(const sym::Vector &v,
                                        const Eigen::VectorXd &bl,
                                        const Eigen::VectorXd &bu) {
  for (size_t i = 0; i < v.size(); ++i) {
    setVariableBounds(v[i], bl[i], bu[i]);
  }
}

void VariableManager::setVariableInitialValue(const sym::Variable &v,
                                              const double &x0) {
  auto it = decision_variable_vec_idx_.find(v.id());
  if (it != decision_variable_vec_idx_.end()) {
  } else {
    std::cout << v << " is not a variable within this program!\n";
    return;
  }
  VariableData &data = decision_variables_data_[it->second];
  data.x0 = x0;
  data.initial_value_updated = true;
}

void VariableManager::setVariableInitialValue(const sym::Vector &v,
                                              const Eigen::VectorXd &x0) {
  for (size_t i = 0; i < v.size(); ++i) {
    setVariableInitialValue(v[i], x0[i]);
  }
}

void VariableManager::listVariables() {
  std::cout << "----------------------\n";
  std::cout << "Variable\n";
  std::cout << "----------------------\n";
  for (sym::Variable &v : decision_variables_) {
    std::cout << v << '\n';
  }
}

}  // namespace symbolic
}  // namespace damotion
