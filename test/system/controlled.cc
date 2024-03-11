#include "system/controlled.h"
#include "pinocchio/parsers/urdf.hpp"

#include <gtest/gtest.h>

TEST(ControlledSystem, Instantiation) { EXPECT_TRUE(false); }

TEST(ControlledSystem, DerivedClass) { EXPECT_TRUE(false); }

TEST(ControlledSystem, FromPinocchioModelWrapper) { 
    
    pinocchio::Model model;
    pinocchio::urdf::buildModel("./ur10_robot.urdf", model, true);
    pinocchio::Data data(model);

    casadi_utils::PinocchioModelWrapper wrapper(model);

    damotion::system::SecondOrderControlledSystem system(wrapper);

    EXPECT_TRUE(true);
    
    }