/**
 * @file fwd.h
 * @author Damian Abood (damian.abood@sydney.edu.au)
 * @brief Forward declarations used for the control module.
 * @version 0.1
 * @date 2024-05-09
 *
 *
 */
#ifndef CONTROL_FWD_H
#define CONTROL_FWD_H

#include <casadi/casadi.hpp>

#include "damotion/optimisation/program.h"
#include "damotion/symbolic/expression.h"

namespace damotion {
namespace control {
namespace opt = damotion::optimisation;
namespace sym = damotion::symbolic;
namespace casadi = ::casadi;
}  // namespace control
}  // namespace damotion

#endif /* CONTROL_FWD_H */
