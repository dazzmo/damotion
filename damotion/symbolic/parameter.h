#ifndef SYMBOLIC_PARAMETER_H
#define SYMBOLIC_PARAMETER_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>
#include <ostream>

namespace damotion {
namespace symbolic {

class Parameter {
 public:
  // ID type for the parameter
  typedef int Id;

  Parameter() = default;

  Parameter(const std::string &name) {
    static int next_id_ = 0;
    next_id_++;
    // Set ID for Parameter
    id_ = next_id_;

    name_ = name;
  }

  ~Parameter() = default;

  // enum class Type : uint8_t { kContinuous };

  const Id &id() const { return id_; }
  const std::string &name() const { return name_; }

  bool operator<(const Parameter &v) const { return id() < v.id(); }
  bool operator==(const Parameter &v) const { return id() == v.id(); }

 private:
  Id id_;
  std::string name_;
};

typedef Eigen::VectorX<Parameter> ParameterVector;
typedef Eigen::MatrixX<Parameter> ParameterMatrix;
typedef std::vector<Eigen::Ref<const ParameterVector>> ParameterRefVector;

// Parameter matrix
ParameterMatrix CreateParameterMatrix(const std::string &name, const int m,
                                      const int n);
// Parameter vector
ParameterVector CreateParameterVector(const std::string &name, const int n);
// Create vector of decision Parameters
ParameterVector ConcatenateParameterRefVector(const ParameterRefVector &vars);

// Operator overloading
std::ostream &operator<<(std::ostream &os, Parameter var);
std::ostream &operator<<(std::ostream &os, ParameterVector vector);
std::ostream &operator<<(std::ostream &os, ParameterMatrix mat);

}  // namespace symbolic
}  // namespace damotion

#endif /* SYMBOLIC_PARAMETER_H */
