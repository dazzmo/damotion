/**
 * @file fwd.h
 * @author Damian Abood (damian.abood@sydney.edu.au)
 * @brief Forward declarations for the optimisation module.
 * @version 0.1
 * @date 2024-05-09
 *
 *
 */
#ifndef OPTIMISATION_FWD_H
#define OPTIMISATION_FWD_H

#include "damotion/symbolic/expression.h"
#include "damotion/symbolic/parameter.h"
#include "damotion/symbolic/variable.h"

namespace damotion {
namespace optimisation {
namespace sym = damotion::symbolic;
// Set true namespace of casadi unless otherwise stated
namespace casadi = ::casadi;
}  // namespace optimisation
}  // namespace damotion

#endif /* OPTIMISATION_FWD_H */