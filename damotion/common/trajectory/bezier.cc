#include "damotion/common/trajectory/bezier.h"

namespace damotion {
namespace trajectory {

std::unique_ptr<Bezier<double>> TruncateBezier(Bezier<double> &curve,
                                               double t) {
  // Perform De Casteljau's algorithm to compute the coefficients
  // New curve
  std::unique_ptr<Bezier<double>> new_curve =
      std::make_unique<Bezier<double>>(curve.dim(), curve.order());

  // Normalise curve to be within [0, 1] to determine break point
  double tau = (t - curve.t0()) / (curve.tf() - curve.t0());

  // Working vector
  int n = curve.order();
  std::vector<Eigen::VectorXd> Q = curve.ControlPoints();
  for (int k = 1; k <= n; ++k) {
    for (int i = 0; i <= n - k; ++i) {
      Q[i] = (1 - tau) * Q[i] + tau * Q[i + 1];
    }
    new_curve->ControlPoints()[k] = Q[0];
  }

  // Update interval the new curve is defined over
  new_curve->UpdateBeginningTime(curve.t0());
  new_curve->UpdateEndingTime(t);

  // Return new curve
  return new_curve;
}

void RandomiseBezier(Bezier<double> &curve, math::RandomNumberGenerator &rng,
                     double gamma) {
  // Create linear vector between points
  Eigen::VectorXd l = (curve.end() - curve.begin());
  // Get length of line
  double d = l.norm();
  // Normalise vector
  l.normalize();

  // Compute coefficients for bezier curve to create straight line
  std::vector<Eigen::VectorXd> P(curve.order() + 1), Pn(curve.order() + 1);

  P[0] = curve.begin();
  P[1] = curve.end();

  // Create control points that would create a straight line
  Pn[0] = P[0];
  for (int n = 2; n <= curve.order(); ++n) {
    for (int i = 1; i <= n; ++i) {
      if (i == n) {
        // Copy final point
        Pn[i] = P[i - 1];
      } else {
        Pn[i] = P[i - 1] * (i / (n + 1)) + P[i] * ((n + 1 - i) / (n + 1));
      }
    }
    // Set new vector
    P = Pn;
  }

  // Sample points randomly within sphere on each point
  for (int i = 2; i < curve.order(); ++i) {
    // Sample random point
    Eigen::VectorXd r = rng.RandomVector(curve.dim(), -1.0, 1.0);
    // Normalise point to be within unit sphere
    r.normalize();
    // Get component of r parallel to line l
    Eigen::VectorXd p = r.dot(l) * l;
    // Get normal component of r normal to line l
    Eigen::VectorXd n = r - p;

    // Randomly place control point within sphere of linear control point
    double sigma = rng.RandomNumber(0, 1);
    curve.ControlPoints()[i] = P[i] + gamma * d * sigma * n;
  }
}

std::ostream &operator<<(std::ostream &os, Bezier<double> &bezier) {
  std::ostringstream oss;
  for (int j = 0; j < bezier.dim(); ++j) {
    for (int i = 0; i <= bezier.order(); ++i) {
      oss << bezier.ControlPoints()[i][j] << '\t';
    }
    oss << '\n';
  }
  return os << oss.str();
}

}  // namespace trajectory
}  // namespace damotion
