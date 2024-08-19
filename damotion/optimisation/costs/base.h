#ifndef COSTS_BASE_H
#define COSTS_BASE_H

#include <casadi/casadi.hpp>

// #include "damotion/casadi/codegen.h"
#include "damotion/casadi/function.hpp"
// #include "damotion/optimisation/fwd.h"

namespace damotion {
namespace optimisation {

/**
 * @brief Generic cost with a single vector input
 *
 */
class Cost : public FunctionBase<1, double> {
 public:
  using SharedPtr = std::shared_ptr<Cost>;
  using UniquePtr = std::unique_ptr<Cost>;

  using Base = FunctionBase<1, double>;

  const std::string &name() const { return name_; }

  Cost(const std::string &name) : Base() {}
  Cost(const Base &f) : Base(f) {}

 private:
  std::string name_ = "";

  /**
   * @brief Creates a unique id for each cost
   *
   * @return int
   */
  int createID() {
    static int next_id = 0;
    int id = next_id;
    next_id++;
    return id;
  }
};

}  // namespace optimisation
}  // namespace damotion

#endif/* COSTS_BASE_H */
