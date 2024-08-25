#include "damotion/symbolic/variable.hpp"
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

std::ostream &operator<<(std::ostream &os,
                         const damotion::symbolic::VariableVector &v) {
  std::ostringstream oss;
  return os << oss.str();
}

bool VariableVector::add(const Variable &var) {
  if (contains(var)) {
    LOG(ERROR) << var << " is already added to program!";
    return false;
  }
  // Add to variable vector to index map
  variable_data_[var.id()] =
      VariableData({.index = sz_++, .initial_value = 0.0, .value = 0.0});
  // Add variable to vector
  variables_.push_back(var);
  return true;
}

bool VariableVector::add(const MatrixRef &var) {
  // Append to our map
  for (Index i = 0; i < var.rows(); ++i) {
    for (Index j = 0; j < var.cols(); ++j) {
      if (add(var(i, j)) == false) return false;
    }
  }
  return true;
}

bool VariableVector::contains(const Variable &var) {
  // Check if variable is already added to the program
  return std::find(variables_.begin(), variables_.end(), var) !=
         variables_.end();
}

bool VariableVector::remove(const Variable &var) {
  // Determine if entry is in the vector
  auto p = std::find(variables_.begin(), variables_.end(), var);
  if (p == variables_.end()) return false;

  // Remove entry from the vector
  variables_.erase(p);
  sz_--;
  return true;
}

bool VariableVector::remove(const MatrixRef &var) {
  for (Index i = 0; i < var.rows(); ++i) {
    for (Index j = 0; j < var.cols(); ++j) {
      if (remove(var(i, j)) == false) return false;
    }
  }
  return true;
}

bool VariableVector::reorder(const VectorRef &var) {
  assert(var.size() == size() && "Incorrect input size!");

  // For each variable, determine its new location
  for (const Variable &v : variables_) {
    int idx = 0;
    for (Index i = 0; i < var.size(); ++i) {
      if (var(idx).id() == v.id()) break;
      idx++;
    }

    // If variable not found, create error
    if (idx >= var.size()) {
      LOG(ERROR) << v << " was not included within the provided reordering";
      return false;
    }

    variable_data_[v.id()].index = idx;
  }

  return true;
}

const VariableVector::Index &VariableVector::getIndex(const Variable &v) const {
  auto it = variable_data_.find(v.id());
  assert(it != variable_data_.end() && "Variable does not exist");
  return it->second.index;
}

VariableVector::IndexVector VariableVector::getIndices(const Vector &v) const {
  IndexVector indices;
  indices.reserve(v.size());
  for (Index i = 0; i < v.size(); ++i) {
    indices.push_back(getIndex(v[i]));
  }
  // Return vector of indices
  return indices;
}

void VariableVector::initialise(const Variable &var, const double &val) {
  auto it = variable_data_.find(var.id());
  assert(it != variable_data_.end() && "Variable does not exist");
  it->second.initial_value = val;
}

void VariableVector::initialise(const Vector &var, const Eigen::VectorXd &val) {
  for (Index i = 0; i < var.size(); ++i) {
    initialise(var[i], val[i]);
  }
}

const double &VariableVector::getInitialValue(const Variable &var) const {
  auto it = variable_data_.find(var.id());
  assert(it != variable_data_.end() && "Variable does not exist");
  return it->second.initial_value;
}

// void VariableVector::setVariableBounds(const Variable &v, const double &bl,
//                                        const double &bu) {
//   auto it = decision_variable_vec_idx_.find(v.id());
//   if (it != decision_variable_vec_idx_.end()) {
//   } else {
//     std::cout << v << " is not a variable within this program!\n";
//     return;
//   }
//   VariableData &data = variables_data_[it->second];
//   data.bl = bl;
//   data.bu = bu;
//   data.bounds_updated = true;
// }

// void VariableVector::setVariableBounds(const Vector &v,
//                                        const Eigen::VectorXd &bl,
//                                        const Eigen::VectorXd &bu) {
//   for (size_t i = 0; i < v.size(); ++i) {
//     setVariableBounds(v[i], bl[i], bu[i]);
//   }
// }

// void VariableVector::setVariableInitialValue(const Variable &v,
//                                              const double &x0) {
//   auto it = decision_variable_vec_idx_.find(v.id());
//   if (it != decision_variable_vec_idx_.end()) {
//   } else {
//     std::cout << v << " is not a variable within this program!\n";
//     return;
//   }
//   VariableData &data = variables_data_[it->second];
//   data.x0 = x0;
//   data.initial_value_updated = true;
// }

// void VariableVector::setVariableInitialValue(const Vector &v,
//                                              const Eigen::VectorXd &x0) {
//   for (size_t i = 0; i < v.size(); ++i) {
//     setVariableInitialValue(v[i], x0[i]);
//   }
// }

}  // namespace symbolic
}  // namespace damotion
