#include "damotion/symbolic/variable.h"
namespace damotion {
namespace symbolic {

// Variable matrix
VariableMatrix CreateVariableMatrix(const std::string &name, const int m,
                                    const int n) {
  VariableMatrix mat(m, n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      mat(i, j) =
          Variable(name + '_' + std::to_string(i) + '_' + std::to_string(j));
    }
  }
  return mat;
}

// Variable vector
VariableVector CreateVariableVector(const std::string &name, const int n) {
  VariableVector vec(n);
  for (int i = 0; i < n; ++i) {
    vec[i] = Variable(name + '_' + std::to_string(i));
  }
  return vec;
}

// Create vector of decision variables
VariableVector ConcatenateVariableRefVector(const VariableRefVector &vars) {
  int sz = 0;
  for (const auto &var : vars) {
    sz += var.size();
  }

  VariableVector vec(sz);
  int cnt = 0;
  for (const auto &var : vars) {
    vec.segment(cnt, var.size()) = var;
    cnt += var.size();
  }

  return vec;
}

// Operator overloading

std::ostream &operator<<(std::ostream &os, damotion::symbolic::Variable var) {
  return os << var.name();
}

std::ostream &operator<<(std::ostream &os,
                         damotion::symbolic::VariableVector vector) {
  std::ostringstream oss;
  for (int i = 0; i < vector.size(); i++) {
    oss << vector[i] << '\n';
  }
  return os << oss.str();
}

std::ostream &operator<<(std::ostream &os,
                         damotion::symbolic::VariableMatrix mat) {
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

}  // namespace symbolic
}  // namespace damotion