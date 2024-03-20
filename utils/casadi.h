#ifndef UTILS_CASADI_H
#define UTILS_CASADI_H

#include <casadi/casadi.hpp>

namespace damotion {
namespace utils {
namespace casadi {
/**
 * @brief Creates the symbolic inputs of a function given the function that
 * possesses named inputs
 *
 * @param f
 * @return ::casadi::SXVector
 */
::casadi::StringVector CreateInputNames(::casadi::Function &f);

}  // namespace casadi
}  // namespace utils
}  // namespace damotion

#endif /* UTILS_CASADI_H */
