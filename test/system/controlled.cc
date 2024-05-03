#include "damotion/system/controlled.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "pinocchio/parsers/urdf.hpp"

TEST(ControlledSystem, Instantiation) { EXPECT_TRUE(false); }

TEST(ControlledSystem, DerivedClass) { EXPECT_TRUE(false); }

TEST(ControlledSystem, FromPinocchioModelWrapper) {
  pinocchio::Model model;
  pinocchio::urdf::buildModel("./ur10_robot.urdf", model, true);
  pinocchio::Data data(model);

  damotion::utils::casadi::PinocchioModelWrapper wrapper(model);

  damotion::system::SecondOrderControlledSystem system(wrapper);

  EXPECT_TRUE(true);
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
