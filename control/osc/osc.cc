#include "control/osc/osc.h"

Eigen::Quaterniond RPYToQuaterion(const double roll, const double pitch,
                                  const double yaw) {
    Eigen::AngleAxisd r = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()),
                      p = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()),
                      y = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    return Eigen::Quaterniond(r * p * y);
}

void OSCController::CreateProgram() {
    damotion::common::Profiler profiler("OSCController::CreateProgram");

    // Create quadratic program data
    QuadraticProgramData data;

    for (auto &p : contact_tasks_) {
        ContactTask &task = p.second;
        // Get index in optimisation variable
        int idx = variable_idx_["lam"] + task.lam_idx;
        // Update conditions on end-effectors
        if (task.inContact) {
            // Update bounds for lambda
            data.ubx.middleRows(idx, task.dim) << 1e8, 1e8, 1e8;
        } else {
            // No contact forces
            data.ubx.middleRows(idx, task.dim).setZero();
        }
    }

    // Update tracking costs
    for (auto &p : tracking_tasks_) {
        TrackingTask &task = p.second;
        Eigen::VectorXd xacc_d = task.ComputeDesiredAcceleration();
        SetParameter(p.first + "_xacc_d", xacc_d);
    }

    // Update costs
    for (auto &p : costs_) {
        Cost &c = p.second;

        // Update cost gradient
        c.g.call();
        data.g += c.w * c.g.getOutput(0);

        // Update cost hessian
        c.H.call();
        data.H += c.w * c.H.getOutput(0);
    }

    // Update constraints
    int idx = 0;
    for (auto &p : constraints_) {
        Constraint &c = p.second;

        // Update constraint jacobian
        c.jac.call();

        // Update entries to data
        data.A.middleRows(idx, c.dim) = c.jac.getOutput(0);
        data.lbA.middleRows(idx, c.dim) = c.lb;
        data.ubA.middleRows(idx, c.dim) = c.ub;

        // Increase index for next entry
        idx += c.dim;
    }

    // Create program

    // Output data to be solved
}

Eigen::VectorXd OSCController::TrackingTask::ComputeDesiredAcceleration() {
    // Evaluate the task with current parameters for the functions
    f.call();
    // Get task position and velocity
    Eigen::VectorXd xpos = f.getOutput(0);
    Eigen::VectorXd xvel = f.getOutput(1);

    if (type == Type::kTranslational) {
        e = xpos - xr;
        de = xvel - vr;
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
        de = Jlog * xvel - wr;
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
        de = Jlog * xvel;
        de.topRows(3) -= vr;     // Translational component
        de.bottomRows(3) -= wr;  // Rotational component
    }

    // Compute desired acceleration
    return Kp * e + Kd * de;
}

OSCController::Cost::Cost(casadi::Function &c, const casadi::SX &x) {
    // TODO: Add name for cost
    // Name of cost
    std::string cost_name = c.name_out(0);
    // Create name vector
    casadi::SXVector in, out;
    inames = {};
    for (int i = 0; i < c.n_in(); ++i) {
        inames.push_back(c.name_in(i));
        in.push_back(casadi::SX::sym(c.name_in(i), c.size1_in(i)));
    }

    // Evaluate
    casadi::SX cost = c(in)[0];

    // Create functions to compute the necessary gradients and hessians
    this->c = eigen::FunctionWrapper(c);

    /* Cost Gradient */
    casadi::Function cg =
        casadi::Function(cost_name + "_grad", in, {gradient(cost, x)}, inames,
                         {cost_name + "_grad"});
    this->g = eigen::FunctionWrapper(cg);

    /* Cost Hessian */
    casadi::Function ch =
        casadi::Function(cost_name + "_hes", in, {hessian(cost, x)}, inames,
                         {cost_name + "_hes"});
    this->H = eigen::FunctionWrapper(ch);
}

OSCController::Constraint::Constraint(casadi::Function &c,
                                      const casadi::SX &x) {
    // Name of cost
    std::string constraint_name = c.name_out(0);
    // Create name vector
    casadi::SXVector in, out;
    inames = {};
    for (int i = 0; i < c.n_in(); ++i) {
        inames.push_back(c.name_in(i));
        in.push_back(casadi::SX::sym(c.name_in(i), c.size1_in(i)));
    }

    // Evaluate
    casadi::SX constraint = c(in)[0];

    // Create functions to compute the necessary gradients and hessians
    this->c = eigen::FunctionWrapper(c);

    /* Constraint Gradient */
    casadi::Function cjac = casadi::Function(constraint_name + "_jac", in,
                                             {jacobian(constraint, x)}, inames,
                                             {constraint_name + "_jac"});
    this->jac = eigen::FunctionWrapper(cjac);

    // Create generic bounds
    ub = std::numeric_limits<double>::infinity() *
         Eigen::VectorXd::Ones(c.size1_out(0));
    lb = -std::numeric_limits<double>::infinity() *
         Eigen::VectorXd::Ones(c.size1_out(0));
}