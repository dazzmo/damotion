#include "control/osc/osc.h"

Eigen::Quaterniond RPYToQuaterion(const double roll, const double pitch,
                                  const double yaw) {
    Eigen::AngleAxisd r = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()),
                      p = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()),
                      y = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    return Eigen::Quaterniond(r * p * y);
}

void OSCController::CreateProgram() {
    // Create projected constraints

    // Construct holonomic constraints

    for (auto &ee : ee_) {
        EndEffectorTrackingData &e = ee.second;
        // Update conditions on end-effectors
        if (e.second.contact) {
            // Add bounds
            lambda_.middleRows(e.lambda_idx, e.lambda_sz).setZero();
        } else {
            // No contact forces
            lambda_.middleRows(e.lambda_idx, e.lambda_sz).setZero();
        }
        // Update tracking costs

        // Add to problem data
    }

    // Add any other costs

    // Create program

    // Output data to be solved
}

void OSCController::TrackingTask::ComputeError() {
    // With current parameters for the functions
    // Evaluate the task
    x.call();
    // Get outputs
    Eigen::VectorXd xpos = x.getOutput(0);
    Eigen::VectorXd xvel = x.getOutput(1);

    if (type == Type::kRotational) {
        // Get rotational component
        Eigen::Quaterniond q(xpos.bottomRows(4));
        // Compute rotational error
        Eigen::Matrix3d R =
            qr.toRotationMatrix().transpose() * q.toRotationMatrix();
        double theta = 0.0;
        e = pinocchio::log3(R, theta);
        // Compute rate of error
        Eigen::Matrix3d Jlog;
        pinocchio::Jlog3(theta, e, Jlog);
        de = Jlog * xvel;
    } else if (type == Type::kTranslational) {

    } else {
    }
}