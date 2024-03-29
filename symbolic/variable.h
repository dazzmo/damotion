#ifndef SOLVERS_VARIABLE_H
#define SOLVERS_VARIABLE_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>
#include <ostream>

namespace damotion {
namespace symbolic {

class Variable {
   public:
    typedef int Id;

    Variable() = default;

    Variable(const std::string &name) {
        static int next_id_ = 0;
        next_id_++;
        // Set ID for variable
        id_ = next_id_;

        name_ = name;
    }

    // enum class Type : uint8_t { kContinuous };

    const Id &id() const { return id_; }
    const std::string &name() const { return name_; }

   private:
    Id id_;
    std::string name_;
};

typedef Eigen::MatrixX<Variable> VariableMatrix;
typedef Eigen::VectorX<Variable> VariableVector;
typedef std::vector<Eigen::Ref<const VariableVector>> VariableRefVector;

// Variable matrix
VariableMatrix CreateVariableMatrix(const std::string &name, const int m,
                                    const int n);
// Variable vector
VariableVector CreateVariableVector(const std::string &name, const int n);
// Create vector of decision variables
VariableVector ConcatenateVariableRefVector(const VariableRefVector &vars);



// class Variable {
//    public:
//     Variable() = default;
//     ~Variable() = default;

//     /**
//      * @brief Create a new variable with name of size (n x m)
//      *
//      * @param name
//      * @param n
//      * @param m
//      */
//     Variable(const std::string &name, int n, int m = 1) : name_(name) {
//         sym_ = casadi::SX::sym(name, n, m);
//         val_ = Eigen::MatrixXd::Zero(n, m);

//         // Set lower and upper bounds
//         lb_ = -std::numeric_limits<double>::infinity() *
//               Eigen::MatrixXd::Ones(n, m);
//         ub_ = std::numeric_limits<double>::infinity() *
//               Eigen::MatrixXd::Ones(n, m);
//     }

//     const std::string &name() { return name_; }

//     /**
//      * @brief Number of rows the variable has
//      *
//      * @return const int&
//      */
//     const int &rows() const { return val_.rows(); }

//     /**
//      * @brief Number of columns the variable has
//      *
//      * @return const int&
//      */
//     const int &cols() const { return val_.cols(); }

//     /**
//      * @brief Symbolic representation of variable
//      *
//      * @return casadi::SX&
//      */
//     casadi::SX &sym() { return sym_; }
//     const casadi::SX &sym() const { return sym_; }

//     /**
//      * @brief Indexing data for the variable within an optimisation vector
//      *
//      * @return const BlockIndex&
//      */
//     BlockIndex &idx() { return idx_; }
//     void SetIndex(const BlockIndex &idx) { idx_ = idx; }

//     /**
//      * @brief Value of the variable
//      *
//      * @return Eigen::MatrixXd&
//      */
//     Eigen::MatrixXd &val() { return val_; }
//     const Eigen::MatrixXd &val() const { return val_; }

//     /**
//      * @brief Number of values within the variable
//      *
//      * @return const int
//      */
//     const int sz() { return val_.size(); }

//     /**
//      * @brief Lower bounds for the variable
//      *
//      * @return Eigen::MatrixXd&
//      */
//     Eigen::MatrixXd &LowerBound() { return lb_; }
//     const Eigen::MatrixXd &LowerBound() const { return lb_; }

//     /**
//      * @brief Upper bounds for the variable
//      *
//      * @return Eigen::MatrixXd&
//      */
//     Eigen::MatrixXd &UpperBound() { return ub_; }
//     const Eigen::MatrixXd &UpperBound() const { return ub_; }

//    private:
//     std::string name_;

//     casadi::SX sym_;
//     Eigen::MatrixXd val_;

//     Eigen::MatrixXd lb_;
//     Eigen::MatrixXd ub_;

//     // Indexing data for the variable
//     BlockIndex idx_;
// };

// Operator overloading
std::ostream &operator<<(std::ostream &os, Variable var);
std::ostream &operator<<(std::ostream &os, VariableVector vector);
std::ostream &operator<<(std::ostream &os, VariableMatrix mat);

}  // namespace symbolic
}  // namespace damotion


#endif /* SOLVERS_VARIABLE_H */
