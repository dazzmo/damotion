#ifndef COSTS_BASE_H
#define COSTS_BASE_H

#include <casadi/casadi.hpp>

#include "damotion/casadi/codegen.h"
#include "damotion/casadi/eigen.h"
#include "damotion/casadi/function.h"
#include "damotion/optimisation/fwd.h"

namespace damotion {
namespace optimisation {

class Cost : public damotion::casadi::CasadiFunction {
 public:
  Cost() = default;
  ~Cost() = default;

  Cost(const int &nx, const int &np = 0)
      : damotion::casadi::CasadiFunction(nx, 1, np) {}

  Cost(const casadi::SX &ex, const casadi::SXVector &x,
       const casadi::SXVector &p, bool grd = false, bool hes = false,
       bool sparse = false)
      : damotion::casadi::CasadiFunction({ex}, x, p, grd, hes, sparse) {}

  /**
   * @brief Name of the cost
   *
   * @return const std::string&
   */
  const std::string &name() const { return name_; }

  /**
   * @brief Set the name of the constraint
   *
   * @param name
   */
  void SetName(const std::string &name) {
    if (name == "") {
      name_ = "cost_" + std::to_string(CreateID());
    } else {
      name_ = name;
    }
  }

 private:
  // Name of the cost
  std::string name_;

  /**
   * @brief Creates a unique id for each cost
   *
   * @return int
   */
  int CreateID() {
    static int next_id = 0;
    int id = next_id;
    next_id++;
    return id;
  }
};

}  // namespace optimisation
}  // namespace damotion

#endif /* COSTS_BASE_H */
