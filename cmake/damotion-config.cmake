include(CMakeFindDependencyMacro)

find_dependency(Boost)
find_dependency(casadi)
find_dependency(Eigen3)
find_dependency(glog)
find_dependency(pinocchio)
find_dependency(qpOASES)

# these are autogenerate by cmake
include("${CMAKE_CURRENT_LIST_DIR}/damotion-targets.cmake")

get_target_property(damotion_INCLUDE_DIRS damotion::damotion INTERFACE_INCLUDE_DIRECTORIES)
list(APPEND damotion_INCLUDE_DIRS ${pinocchio_INCLUDE_DIRS})

get_property(damotion_LIBRARIES TARGET damotion::damotion PROPERTY LOCATION)
list(APPEND damotion_LIBRARIES ${pinocchio_LIBRARIES} casadi glog)