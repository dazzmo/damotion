#ifndef SOLVERS_VARIABLE_H
#define SOLVERS_VARIABLE_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>
#include <ostream>

namespace damotion {
namespace symbolic {

class Variable {
   public:
    // ID type for the variables
    typedef int Id;

    Variable() = default;

    Variable(const std::string &name) {
        static int next_id_ = 0;
        next_id_++;
        // Set ID for variable
        id_ = next_id_;

        name_ = name;
    }

    ~Variable() = default;

    // enum class Type : uint8_t { kContinuous };

    const Id &id() const { return id_; }
    const std::string &name() const { return name_; }

    bool operator<(const Variable &v) const { return id() < v.id(); }
    bool operator==(const Variable &v) const { return id() == v.id(); }

   private:
    Id id_;
    std::string name_;
};

typedef Eigen::MatrixX<Variable> VariableMatrix;
typedef Eigen::VectorX<Variable> VariableVector;
typedef std::shared_ptr<VariableVector> VariableVectorSharedPtr;
typedef std::vector<Eigen::Ref<const VariableVector>> VariableRefVector;

// Variable matrix
VariableMatrix CreateVariableMatrix(const std::string &name, const int m,
                                    const int n);
// Variable vector
VariableVector CreateVariableVector(const std::string &name, const int n);
// Create vector of decision variables
VariableVector ConcatenateVariableRefVector(const VariableRefVector &vars);

// Operator overloading
std::ostream &operator<<(std::ostream &os, Variable var);
std::ostream &operator<<(std::ostream &os, VariableVector vector);
std::ostream &operator<<(std::ostream &os, VariableMatrix mat);

}  // namespace symbolic
}  // namespace damotion

#endif /* SOLVERS_VARIABLE_H */
