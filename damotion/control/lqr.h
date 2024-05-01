#ifndef LQR_H
#define LQR_H

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

/**
 * @brief Solution to the Discrete Algebraic Riccatti equation
 * Aᵀ P + P A - P B R⁻¹ Bᵀ P + Q = 0
 *
 */
Eigen::MatrixXd care(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                     const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R) {
  // Create Hamiltonian
  Eigen::MatrixXd Z(A.rows() + Q.rows(), A.cols() + A.rows());
  Z << A, -B * R.inverse() * B.transpose(), -Q, -A.transpose();
  // Perform Schur decomposition
  Eigen::RealSchur<Eigen::Matrix> schur(Z);

  Eigen::MatrixXd U = schur.matrixU();
  Eigen::MatrixXd T = schur.matrixT();

  Eigen::MatrixXd P = U.bottomRows(A.rows()) * U.topRows(A.rows()).inverse();

  return P;
}

/**
 * @brief Solution to the Discrete Algebraic Riccatti equation
 * Aᵀ P + P A - P B R⁻¹ Bᵀ P + Q = 0
 *
 */
Eigen::MatrixXd dare(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                     const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R) {
  // Iterate backwards in time
  Eigen::MatrixXd P = Q;
  int time_steps = 10;

  for (int i = 0; i < time_steps; ++i) {
    P = Q + A.transpose() * P * A -
        A.transpose() * P * B * (B.transpose() * P * B + R).inverse() *
            B.transpose() * P * A;
    // Check convergence
  }

  // Compute control
  Eigen::MatrixXd K =
      (R + B.transpose() * P * B).inverse() * B.transpose() * P * A;

  return Eigen::MatrixXd::Zeros(1, 1);
}

#endif /* LQR_H */
