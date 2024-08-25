#ifndef SYMBOLIC_VARIABLE_H
#define SYMBOLIC_VARIABLE_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>
#include <ostream>

#include "damotion/optimisation/bounds.hpp"
#include "damotion/optimisation/initial_value.hpp"

namespace damotion {
namespace symbolic {

class Variable {
 public:
  // ID type for the variables
  typedef std::size_t Id;

  enum class Type { CONTINUOUS = 0, DISCRETE };

  Variable() = default;

  Variable(const std::string &name) : name_(name) {
    static int next_id_ = Id(0);
    id_ = next_id_++;
  }

  ~Variable() = default;

  const Id &id() const { return id_; }
  const std::string &name() const { return name_; }

  bool operator<(const Variable &v) const { return id() < v.id(); }
  bool operator==(const Variable &v) const { return id() == v.id(); }

 private:
  Id id_;
  std::string name_;
};

typedef Eigen::VectorX<Variable> Vector;
typedef Eigen::MatrixX<Variable> Matrix;

typedef Eigen::Ref<const Vector> VectorRef;
typedef Eigen::Ref<const Matrix> MatrixRef;

typedef std::list<VectorRef> VectorRefList;
typedef std::list<VectorRef> MatrixRefList;

// Variable matrix
Matrix createMatrix(const std::string &name, const int m, const int n);
// Variable vector
Vector createVector(const std::string &name, const int n);

// Operator overloading
std::ostream &operator<<(std::ostream &os, Variable var);
std::ostream &operator<<(std::ostream &os, Vector vector);
std::ostream &operator<<(std::ostream &os, Matrix mat);

/**
 * @brief Class that maintains and adjusts variables organised into a
 * vector
 *
 */
class VariableVector {
 public:
  using Index = Eigen::Index;
  using IndexVector = std::vector<Index>;

  using SharedPtr = std::shared_ptr<VariableVector>;
  using UniquePtr = std::unique_ptr<VariableVector>;

  VariableVector() : sz_(0) {}

  ~VariableVector() = default;

  /**
   * @brief Number of variables comprising the vector
   *
   * @return const int&
   */
  const Index &size() const { return sz_; }

  /**
   * @brief Adds a decision variable/ returns true if added, false if it is
   * already included.
   *
   * @param var
   */
  bool add(const Variable &var);

  /**
   * @brief Returns the vector of all variables within the variable vector.
   *
   * @return const std::vector<Variable>&
   */
  const std::vector<Variable> &all() const { return variables_; }

  /**
   * @brief Add decision variables to the vector. Returns true if added, false
   * if it is already included.
   *
   * @param var
   */
  bool add(const MatrixRef &var);

  /**
   * @brief Removes variable currently considered by the program. True on
   * success and false otherwise.
   *
   * @param var
   */
  bool remove(const Variable &var);

  /**
   * @brief Removes multiple variables currently considered by the program. True
   * on success and false otherwise.
   *
   * @param var
   */
  bool remove(const MatrixRef &var);

  /**
   * @brief Whether a variable var is contained within the vector
   *
   * @param var
   * @return true
   * @return false
   */
  bool contains(const Variable &var);

  /**
   * @brief Initialise the variables with the value val.
   *
   * @param var
   * @param val
   */
  void initialise(const Variable &var, const double &val);

  void initialise(const Vector &var, const Eigen::VectorXd &val);

  const double &getInitialValue(const Variable &var) const;

  /**
   * @brief Returns the index of the given variable within the created
   * optimisation vector
   *
   * @param v
   * @return int
   */
  const Index &getIndex(const Variable &v) const;

  /**
   * @brief Returns a vector of indices for the position of each entry in v in
   * the current variable vector.
   *
   * @param v
   * @return IndexVector
   */
  IndexVector getIndices(const Vector &v) const;

  /**
   * @brief Sets the optimisation vector using the given ordering of variables
   *
   * @param var
   */
  bool reorder(const VectorRef &var);

  /**
   * @brief Prints the current set of parameters for the program to the
   * screen
   *
   */
  void list();

 private:
  // Number of decision variables
  Index sz_;

  Eigen::VectorXd initial_value_;
  Eigen::VectorXd value_;

  struct VariableData {
    Index index;
    double initial_value;
    double value;
  };

  // Location of each decision variable within the optimisation vector
  std::unordered_map<Variable::Id, VariableData> variable_data_;
  // Vector of all decision variables used
  std::vector<Variable> variables_;
};

std::ostream &operator<<(std::ostream &os, const VariableVector &vec);

}  // namespace symbolic
}  // namespace damotion

#endif /* SYMBOLIC_VARIABLE_H */
