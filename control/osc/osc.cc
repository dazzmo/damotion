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

    // for (auto &ee : ee_) {
    //     EndEffectorTrackingData &e = ee.second;
    //     // Update conditions on end-effectors
    //     if (e.second.contact) {
    //         // Add bounds
    //         lambda_.middleRows(e.lambda_idx, e.lambda_sz).setZero();
    //     } else {
    //         // No contact forces
    //         lambda_.middleRows(e.lambda_idx, e.lambda_sz).setZero();
    //     }
    //     // Update tracking costs

    //     // Add to problem data
    // }

    // Add any other costs

    // Create program

    // Output data to be solved
}

void OSCController::TrackingTask::UpdateTrackingError() {
    // Evaluate the task with current parameters for the functions
    x.call();
    // Get task position and velocity
    Eigen::VectorXd xpos = x.getOutput(0);
    Eigen::VectorXd xvel = x.getOutput(1);

    if (type == Type::kTranslational) {
        e = xpos - xr;
        de = xvel;  // ! Currently set at zero velocity
    } else if (type == Type::kRotational) {
        // Get rotational component
        Eigen::Vector4d qv = xpos.bottomRows(4);
        Eigen::Quaterniond q(qv);
        // Compute rotational error
        Eigen::Matrix3d R =
            qr.toRotationMatrix().transpose() * q.toRotationMatrix();
        double theta = 0.0;
        Eigen::Vector3d elog;
        elog = pinocchio::log3(R, theta);
        // Compute rate of error
        Eigen::Matrix3d Jlog;
        pinocchio::Jlog3(theta, elog, Jlog);
        e = elog;
        de = Jlog * xvel;  // ! Currently set at zero velocity
    } else {
        // Convert to 6D pose
        Eigen::Vector4d qv = xpos.bottomRows(4);
        Eigen::Quaterniond q(qv);
        Eigen::Vector3d x = xpos.topRows(3);
        pinocchio::SE3 se3r(qr.toRotationMatrix(), xr),
            se3(q.toRotationMatrix(), x);

        e = pinocchio::log6(se3r.actInv(se3)).toVector();
        Eigen::Matrix<double, 6, 6> Jlog;
        pinocchio::Jlog6(se3r.actInv(se3), Jlog);
        de = Jlog * xvel;  // ! Currently set at zero velocity
    }
}

OSCController::Cost::Cost(casadi::Function &c) {
    // Create name vector
    inames = {};
    for (int i = 0; i < c.n_in(); ++i) {
        inames.push_back(c.name_in(i));
    }

    std::string cost_name = c.name_out(0);

    // Create functions to compute the necessary gradients and hessians
    this->c = eigen::FunctionWrapper(c);

    /* Cost Gradient */
    // TODO: Make gradient in terms of optimsiation vector x = [qacc, ctrl, lam]
    casadi::Function cg =
        c.factory(c.name() + "_grad", inames,
                  {"grad:" + cost_name + ":qacc", "grad:" + cost_name + ":ctrl",
                   "grad:" + cost_name + ":lam"});
    this->g = eigen::FunctionWrapper(cg);

    /* Cost Hessian */
    casadi::Function ch =
        c.factory(c.name() + "_hes", inames,
                  {"hess:" + cost_name + ":qacc:qacc", "hess:" + cost_name + ":ctrl:ctrl",
                   "hess:" + cost_name + ":lam:lam"});
    this->H = eigen::FunctionWrapper(ch);
}