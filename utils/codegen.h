#ifndef UTILS_CODEGEN_H
#define UTILS_CODEGEN_H

#include <dlfcn.h>
#include <filesystem>

#include <casadi/casadi.hpp>


namespace casadi_utils {

/**
 * @brief Generates a dynamically linkable library for the function f and loads
 * the binary into code. Returns a function which uses the library.
 *
 * @param f Function to perform code generation for
 * @param dir Directory to store the binary
 * @return casadi::Function Function that utilises the created and loaded
 * library
 */
casadi::Function codegen(const casadi::Function &f,
                         const std::string &dir = "./");

}  // namespace casadi_utils

#endif /* UTILS_CODEGEN_H */
