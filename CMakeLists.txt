cmake_minimum_required(VERSION 3.20)
project(damotion VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/)  

include_directories(/usr/local/include)
link_directories(/usr/local/lib)


find_package(Boost REQUIRED)
find_package(casadi REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(glog REQUIRED)
find_package(pinocchio REQUIRED)

SET(${PROJECT_NAME}_HEADERS
common/math/binomial.h
common/math/random.h
common/trajectory/bezier.h
common/trajectory/piecewise_bezier.h
common/trajectory/trajectory.h
common/profiler.h
control/osc/osc.h
control/osc/tasks/task.h
control/osc/tasks/motion.h
control/osc/tasks/contact.h
control/lqr.h
estimation/ekf.h
model/frame.h
solvers/binding.h
solvers/bounds.h
solvers/constraint.h
solvers/cost.h
solvers/program.h
solvers/solver.h
symbolic/expression.h
symbolic/parameter.h
symbolic/variable.h
system/controlled.h
system/system.h
utils/casadi.h
utils/codegen.h
utils/eigen_wrapper.h
utils/log.h
utils/pinocchio_model.h
)

# Collect source files
SET(${PROJECT_NAME}_SOURCES
common/math/binomial.cc
common/trajectory/bezier.cc
common/trajectory/trajectory.cc
control/osc/osc.cc
control/osc/tasks/motion.cc
control/osc/tasks/contact.cc
solvers/bounds.cc
solvers/constraint.cc
solvers/cost.cc
solvers/program.cc
solvers/solver.cc
symbolic/variable.cc
system/controlled.cc
system/system.cc
utils/casadi.cc
utils/codegen.cc
utils/eigen_wrapper.cc
utils/log.cc
utils/pinocchio_model.cc
)

SET(${PROJECT_NAME}_LIBRARIES
Eigen3::Eigen 
pinocchio::pinocchio 
casadi
glog 
)

if(WITH_QPOASES)
find_package(qpOASES REQUIRED)
list(APPEND ${PROJECT_NAME}_LIBRARIES qpOASES)
list(APPEND ${PROJECT_NAME}_HEADERS solvers/solve_qpoases.h)
endif(WITH_QPOASES)

if(WITH_IPOPT)
list(APPEND ${PROJECT_NAME}_LIBRARIES ipopt)
list(APPEND ${PROJECT_NAME}_SOURCES solvers/solve_ipopt.cc)
list(APPEND ${PROJECT_NAME}_HEADERS solvers/solve_ipopt.h)
endif(WITH_IPOPT)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(${PROJECT_NAME} 
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>"
)

target_compile_definitions(${PROJECT_NAME} PRIVATE "-DINCLUDE_DIR=\"${PROJECT_SOURCE_DIR}\"")

if(BUILD_TESTS)
  add_subdirectory(test)
endif(BUILD_TESTS)

# Install
include(GNUInstallDirs) # for correct libraries locations across platforms
set(config_package_location "share/${PROJECT_NAME}/cmake") # for .cmake find-scripts installs
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
)
# Copy header files
install(DIRECTORY 
        "${CMAKE_SOURCE_DIR}/common"
        "${CMAKE_SOURCE_DIR}/control"
        "${CMAKE_SOURCE_DIR}/planning"
        "${CMAKE_SOURCE_DIR}/model"
        "${CMAKE_SOURCE_DIR}/solvers"
        "${CMAKE_SOURCE_DIR}/symbolic"
        "${CMAKE_SOURCE_DIR}/system"
        "${CMAKE_SOURCE_DIR}/utils"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME} # target directory
        FILES_MATCHING # install only matched files
        PATTERN "*.h" # select header files
)

# Auto generate and install the find scripts helpers
install(
  EXPORT ${PROJECT_NAME}-targets
  FILE ${PROJECT_NAME}-targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${config_package_location}
)
# generate the file damotionConfigVersion.cmake
set(version_file "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake")
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${version_file}
  VERSION ${damotion_VERSION}
  COMPATIBILITY SameMajorVersion
)
# install the two files in a place where cmake looks for them
install(
  FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake"   # self-written
    ${version_file}                                         # generated
  DESTINATION ${config_package_location}
)