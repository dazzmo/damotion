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

void VariableVector::add(const Variable &var) {
  if (!contains(var)) {
    // Add to variable vector
    variable_idx_[var.id()] = sz_;
    variables_.push_back(var);
    // Increase count of decision variables
    sz_++;
    lb().conservativeResize(sz_);
    ub().conservativeResize(sz_);
    initialValue().conservativeResize(sz_);
  } else {
    // Variable already added to program!
    std::cout << var << " is already added to program!\n";
  }
}

void VariableVector::add(const MatrixRef &var) {
  // Append to our map
  for (Index i = 0; i < var.rows(); ++i) {
    for (Index j = 0; j < var.cols(); ++j) {
      add(var(i, j));
    }
  }
}

bool VariableVector::contains(const Variable &var) {
  // Check if variable is already added to the program
  return std::find(variables_.begin(), variables_.end(), var) !=
         variables_.end();
}

bool VariableVector::reorder(const VectorRef &var) {
  assert(var.size() == size() && "Incorrect input size!");

  // Set indices by order in the decision variable vector
  for (const Variable &v : variables_) {
    // Find starting point of v within var
    int idx = 0;
    Variable::Id id = v.id();
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
      variable_idx_[id] = idx;
    }
  }

  return true;
}

void VariableVector::remove(const MatrixRef &var) {
  // TODO - Implement
}

const VariableVector::Index &VariableVector::getIndex(const Variable &v) const {
  auto it = variable_idx_.find(v.id());
  assert(it == variable_idx_.end() && "Variable does not exist");
  return it->second;
}

VariableVector::IndexVector VariableVector::getIndices(const Vector &v) {
  IndexVector indices;
  indices.reserve(v.size());
  for (Index i = 0; i < v.size(); ++i) {
    Index idx = getIndex(v[i]);
    if (idx >= 0) {
      indices.push_back(idx);
    }
  }
  // Return vector of indices
  return indices;
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

void VariableVector::list() {
  std::cout << "----------------------\n";
  std::cout << "Variable\n";
  std::cout << "----------------------\n";
  for (Variable &v : variables_) {
    std::cout << v << '\n';
  }
}

}  // namespace symbolic
}  // namespace damotion
