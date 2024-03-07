include(CMakeFindDependencyMacro)

find_dependency(pinocchio)
find_dependency(casadi)
find_dependency(Eigen3)
find_dependency(glog)
find_dependency(qpOASES)

# these are autogenerate by cmake
include("${CMAKE_CURRENT_LIST_DIR}/casadi_project-targets.cmake")

get_target_property(casadi_project_INCLUDE_DIRS casadi_project::casadi_project INTERFACE_INCLUDE_DIRECTORIES)
list(APPEND casadi_project_INCLUDE_DIRS ${pinocchio_INCLUDE_DIRS})

get_property(casadi_project_LIBRARIES TARGET casadi_project::casadi_project PROPERTY LOCATION)
list(APPEND casadi_project_LIBRARIES ${pinocchio_LIBRARIES} casadi glog)