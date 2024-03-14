#include "common/math/binomial.h"

namespace damotion {
namespace math {

double BinomialCoefficient(int n, int k) {
    if (k > n) {
        // TODO: Throw error
    }
    if (k == 0) {
        return 1.0;
    }
    if (k < n / 2) {
        return BinomialCoefficient(n, n - k);
    }
    return n * BinomialCoefficient(n - 1, k - 1) / k;
}

}  // namespace math
}  // namespace damotion