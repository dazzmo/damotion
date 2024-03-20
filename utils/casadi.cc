#include "utils/casadi.h"

namespace damotion {
namespace utils {
namespace casadi {

::casadi::StringVector CreateInputNames(::casadi::Function &f) {
    ::casadi::StringVector in;
    for (int i = 0; i < f.n_in(); ++i) {
        // Add inputs to constraint
        in.push_back(f.name_in(i));
    }
    return in;
}

}  // namespace casadi
}  // namespace utils
}  // namespace damotion