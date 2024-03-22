#ifndef SOLVERS_VARIABLE_H
#define SOLVERS_VARIABLE_H

#include <Eigen/Core>
#include <casadi/casadi.hpp>

namespace damotion {
namespace optimisation {

class Variable {
   public:
    Variable() = default;
    ~Variable() = default;

    /**
     * @brief Create a new variable with name of size (n x m)
     *
     * @param name
     * @param n
     * @param m
     */
    Variable(const std::string &name, int n, int m = 1) : name_(name) {
        sym_ = casadi::SX::sym(name, n, m);
        val_ = Eigen::MatrixXd::Zero(n, m);

        // Set lower and upper bounds
        lb_ = -std::numeric_limits<double>::infinity() *
              Eigen::MatrixXd::Ones(n, m);
        ub_ = std::numeric_limits<double>::infinity() *
              Eigen::MatrixXd::Ones(n, m);
    }

    const std::string &name() { return name_; }

    const int &rows() const {return val_.rows();}
    const int &cols() const {return val_.cols();}

    casadi::SX &sym() { return sym_; }
    const casadi::SX &sym() const { return sym_; }

    Eigen::MatrixXd &val() { return val_; }
    const Eigen::MatrixXd &val() const { return val_; }

    const int sz() { return val_.size(); }

    Eigen::MatrixXd &lb() { return lb_; }
    const Eigen::MatrixXd &lb() const { return lb_; }

    Eigen::MatrixXd &ub() { return ub_; }
    const Eigen::MatrixXd &ub() const { return ub_; }

   private:
    std::string name_;

    casadi::SX sym_;
    Eigen::MatrixXd val_;

    Eigen::MatrixXd lb_;
    Eigen::MatrixXd ub_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_VARIABLE_H */
