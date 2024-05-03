find_package(Boost REQUIRED)
find_package(casadi REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(glog REQUIRED)
find_package(pinocchio REQUIRED)

option(WITH_QPOASES "Build with qpOASES functionality" ON)

# Collect source files
SET(${PROJECT_NAME}_SOURCES
solvers/sparse.cc
solvers/solver.cc
solvers/ipopt.cc
utils/eigen_wrapper.cc
utils/log.cc
utils/codegen.cc
utils/pinocchio_model.cc
optimisation/program.cc
optimisation/bounds.cc
common/function.cc
common/math/binomial.cc
common/trajectory/bezier.cc
common/trajectory/trajectory.cc
common/sparsity.cc
model/frame.cc
symbolic/parameter.cc
symbolic/variable.cc
system/controlled.cc
system/system.cc
control/osc/tasks/contact.cc
control/osc/tasks/motion.cc
control/osc/osc.cc
)

SET(${PROJECT_NAME}_LIBRARIES
Eigen3::Eigen
pinocchio::pinocchio
casadi
glog
)

if(WITH_QPOASES)
  message(STATUS "Using qpOASES as numeric optimization solver")
  find_package(qpOASES REQUIRED)
  list(APPEND ${PROJECT_NAME}_LIBRARIES qpOASES)
  add_compile_definitions(WITH_QPOASES)
endif(WITH_QPOASES)

# if(WITH_IPOPT)
# list(APPEND ${PROJECT_NAME}_LIBRARIES ipopt)
# list(APPEND ${PROJECT_NAME}_SOURCES solvers/solve_ipopt.cc)
# list(APPEND ${PROJECT_NAME}_HEADERS solvers/solve_ipopt.h)
# endif(WITH_IPOPT)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

message("CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR}")

# Jesse: not actually sure what it does, since we dont have a include folder do we need the INSTALL_INTERFACE
# dont know about the build interface
target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
    # "$<INSTALL_INTERFACE:include>"
)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-unused-parameter -pipe)

add_library(damotion::damotion ALIAS ${PROJECT_NAME})

include(GNUInstallDirs)

## First of all, we need to install the library itself.
install(TARGETS ${PROJECT_NAME}
    EXPORT damotion-export
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    # Perhaps use this, instead of installing include dir
    #PUBLIC_HEADER DESTINATION include
    INCLUDES DESTINATION include # We need this right?
    RUNTIME DESTINATION bin
)

# Install header files
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)
