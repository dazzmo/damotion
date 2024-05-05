#include "damotion/symbolic/parameter.h"
namespace damotion {
namespace symbolic {

// Parameter matrix
ParameterMatrix CreateParameterMatrix(const std::string &name, const int m,
                                      const int n) {
  ParameterMatrix mat(m, n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      mat(i, j) =
          Parameter(name + '_' + std::to_string(i) + '_' + std::to_string(j));
    }
  }
  return mat;
}

// Parameter vector
ParameterVector CreateParameterVector(const std::string &name, const int n) {
  ParameterVector vec(n);
  for (int i = 0; i < n; ++i) {
    vec[i] = Parameter(name + '_' + std::to_string(i));
  }
  return vec;
}

// Create vector of decision Parameters
ParameterVector ConcatenateParameterRefVector(const ParameterRefVector &vars) {
  int sz = 0;
  for (const auto &var : vars) {
    sz += var.size();
  }

  ParameterVector vec(sz);
  int cnt = 0;
  for (const auto &var : vars) {
    vec.segment(cnt, var.size()) = var;
    cnt += var.size();
  }

  return vec;
}

// Operator overloading

std::ostream &operator<<(std::ostream &os, damotion::symbolic::Parameter var) {
  return os << var.name();
}

std::ostream &operator<<(std::ostream &os,
                         damotion::symbolic::ParameterVector vector) {
  std::ostringstream oss;
  for (int i = 0; i < vector.size(); i++) {
    oss << vector[i] << '\n';
  }
  return os << oss.str();
}

std::ostream &operator<<(std::ostream &os,
                         damotion::symbolic::ParameterMatrix mat) {
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
