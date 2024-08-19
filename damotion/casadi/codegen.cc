#include "damotion/casadi/codegen.h"

namespace damotion {
namespace casadi {

::casadi::Function codegen(const ::casadi::Function &f,
                           const std::string &dir) {
  // Create binary in desired directory
  // Get current path
  auto path = std::filesystem::current_path();
  // Change to new path
  std::filesystem::current_path(dir);

  // TODO - Throw warning if it doesn't exist

  // Create hash
  std::string f_str = f.serialize();
  std::size_t hash = std::hash<std::string>()(f_str);

  // Create new name
  std::string f_name = f.name() + "_codegen_" + std::to_string(hash);

  // Check if file exists
  if (!std::filesystem::exists(f_name + ".so")) {
    // If binary doesn't exist, create it
    f.generate(f_name + ".c");
    std::cout << f_name + " not found. Compiling...\n";
    int ret = system(("gcc -fPIC -shared -O3 -march=native " + f_name +
                      ".c -o " + f_name + ".so")
                         .c_str());
  }

  // Load the binary
  std::cout << "Found " + f_name + ". Loading...\n";
  void *handle;
  handle = dlopen(("./" + f_name + ".so").c_str(), RTLD_NOW);
  if (handle == 0) {
    std::cout << "Cannot open" + f_name + ": " << dlerror() << '\n';
    // Return back to normal path
    std::filesystem::current_path(path);
    return f;
  }

  // Load the generated function
  ::casadi::Function fcg = ::casadi::external(f.name(), "./" + f_name + ".so");

  // Return back to normal path
  std::filesystem::current_path(path);

  return fcg;
}

}  // namespace casadi
}  // namespace damotion
