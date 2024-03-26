#ifndef SOLVERS_BINDING_H
#define SOLVERS_BINDING_H

#include "solvers/constraint.h"
#include "solvers/variable.h"

namespace damotion {
namespace optimisation {

class CostBinding {
   public:
   private:
};

class ConstraintBinding {
   public:
    void UpdateConstraint(const Eigen::VectorXd &x, const Eigen::VectorXd &p) {
        // Update constraint with optimisation and parameter vector
        int cnt = 0;
        for (const int &i : x_idx_) {
            c_.ConstraintFunction().setInput(cnt, x.data() + i);
        }
        for (const int &i : p_idx_) {
            c_.ConstraintFunction().setInput(cnt, p.data() + i);
        }
        c_.ConstraintFunction().call();
    }

   private:
    Constraint &c_;
    // Starting indices of variables within the optimisation vector
    std::vector<int> x_idx_;
    // Starting indices of parameters within the parameter vector
    std::vector<int> p_idx_;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_BINDING_H */
