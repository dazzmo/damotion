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
  typedef size_t Id;

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
// Create vector of decision variables
Vector concatenateVariables(const VectorRefList &vars);
Vector concatenateVariables(const MatrixRefList &vars);

// Operator overloading
std::ostream &operator<<(std::ostream &os, Variable var);
std::ostream &operator<<(std::ostream &os, Vector vector);
std::ostream &operator<<(std::ostream &os, Matrix mat);

/**
 * @brief Class that maintains and adjusts variables organised into a
 * vector
 *
 */
class VariableManager {
 public:
  using SharedPtr = std::shared_ptr<VariableManager>;
  using UniquePtr = std::unique_ptr<VariableManager>;

  VariableManager() : n_variables_(0) {}
  ~VariableManager() = default;

  /**
   * @brief Number of decision variables currently in the program
   *
   * @return const int&
   */
  const int &numberOfVariables() const { return n_variables_; }

  /**
   * @brief Adds a decision variable
   *
   * @param var
   */
  void addVariable(const Variable &var);

  /**
   * @brief Add decision variables
   *
   * @param var
   */
  void addVariables(const MatrixRef &var);

  /**
   * @brief Removes variables currently considered by the program.
   *
   * @param var
   */
  void removeVariables(const MatrixRef &var);

  /**
   * @brief Whether a variable var is a decision variable within the program
   *
   * @param var
   * @return true
   * @return false
   */
  bool isVariable(const Variable &var);

  /**
   * @brief Returns the index of the given variable within the created
   * optimisation vector
   *
   * @param v
   * @return int
   */
  int getVariableIndex(const Variable &v);

  /**
   * @brief Returns a vector of indices for the position of each entry in v in
   * the current decision variable vector.
   *
   * @param v
   * @return std::vector<int>
   */
  std::vector<int> getVariableIndices(const Vector &v);

  /**
   * @brief Set the vector of decision variables to the default ordering of
   * variables (ordered by when they were added)
   *
   */
  void setVector();

  /**
   * @brief Sets the optimisation vector with the given ordering of variables
   *
   * @param var
   */
  bool setVector(const VectorRef &var);

  /**
   * @brief Returns a vector of the values of each variable entry in the manager
   * in the current order provided
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &getVariableValueVector() const { return x_; }

  /**
   * @brief Determines whether a vector of variables var is continuous within
   * the optimisation vector of the program.
   *
   * @param var
   * @return true
   * @return false
   */
  bool isContinuousInVector(const Vector &var);

  /**
   * @brief Updates the decision variable bound vectors with all the current
   * values set for the decision variables.
   *
   */
  void updateVariableBoundVectors() {
    for (size_t i = 0; i < decision_variables_.size(); ++i) {
      VariableData &data = decision_variables_data_[i];
      if (data.bounds_updated) {
        int idx = getVariableIndex(decision_variables_[i]);
        xbl_[idx] = data.bl;
        xbu_[idx] = data.bu;
      }
    }
  }

  /**
   * @brief Updates the initial value vector for the decision variables with all
   * the current values set for the decision variables.
   *
   */
  void updateInitialValueVector() {
    for (size_t i = 0; i < decision_variables_.size(); ++i) {
      VariableData &data = decision_variables_data_[i];
      if (data.initial_value_updated) {
        int idx = getVariableIndex(decision_variables_[i]);
        x0_[idx] = data.x0;
      }
    }
  }

  /**
   * @brief Vector of initial values for the decision variables within the
   * program.
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &variableInitialValues() const { return x0_; }

  /**
   * @brief Upper bound for decision variables within the current program.
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &variableupperBounds() const { return xbu_; }

  /**
   * @brief Upper bound for decision variables within the current program.
   *
   * @return const Eigen::VectorXd&
   */
  const Eigen::VectorXd &variablelowerBounds() const { return xbl_; }

  void setVariableBounds(const Variable &v, const double &lb, const double &ub);
  void setVariableBounds(const Vector &v, const Eigen::VectorXd &lb,
                         const Eigen::VectorXd &ub);

  void setVariableInitialValue(const Variable &v, const double &x0);
  void setVariableInitialValue(const Vector &v, const Eigen::VectorXd &x0);

  /**
   * @brief Prints the current set of parameters for the program to the
   * screen
   *
   */
  void listVariables();

 private:
  // Number of decision variables
  int n_variables_;

  // Decision variable upper bounds
  Eigen::VectorXd xbu_;
  // Decision variable lower bounds
  Eigen::VectorXd xbl_;
  // Initial values for decision variables
  Eigen::VectorXd x0_;
  // Vector of variable values
  Eigen::VectorXd x_;

  // Location of each decision variable within the optimisation vector
  std::unordered_map<Variable::Id, int> decision_variable_idx_;
  // Index locations for data related to each variable
  std::unordered_map<Variable::Id, int> decision_variable_vec_idx_;
  // Vector of all decision variables used
  std::vector<Variable> decision_variables_;

  struct VariableData {
    bool bounds_updated = false;
    bool initial_value_updated = false;
    double bl = -std::numeric_limits<double>::infinity();
    double bu = std::numeric_limits<double>::infinity();
    double x0 = 0.0;
  };

  std::vector<VariableData> decision_variables_data_;
};

}  // namespace symbolic
}  // namespace damotion

#endif /* SOLVERS_VARIABLE_H */
