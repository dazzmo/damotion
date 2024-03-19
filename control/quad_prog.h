#ifndef CONTROL_QUAD_PROG_H
#define CONTROL_QUAD_PROG_H

#include <Eigen/Core>

struct QuadraticProgramData {
    Eigen::MatrixXd H;
    Eigen::VectorXd g;

    Eigen::MatrixXd A;
    Eigen::VectorXd ubA;
    Eigen::VectorXd lbA;

    // Variable upper bound
    Eigen::VectorXd ubx;
    // Variable lower bound
    Eigen::VectorXd lbx;
};

#endif/* CONTROL_QUAD_PROG_H */
