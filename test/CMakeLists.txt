find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
link_directories(${GTEST_LIB_DIRS})

enable_testing()

# Utils
add_executable(codegen_test ${CMAKE_CURRENT_SOURCE_DIR}/utils/codegen.cc)
target_link_libraries(codegen_test damotion GTest::gtest_main)

add_executable(eigen_wrapper_test ${CMAKE_CURRENT_SOURCE_DIR}/utils/eigen_wrapper.cc)
target_link_libraries(eigen_wrapper_test damotion GTest::gtest_main)

add_executable(pinocchio_wrapper_test ${CMAKE_CURRENT_SOURCE_DIR}/utils/pinocchio_model.cc)
target_link_libraries(pinocchio_wrapper_test damotion GTest::gtest_main)

add_executable(log ${CMAKE_CURRENT_SOURCE_DIR}/utils/log.cc)
target_link_libraries(log damotion GTest::gtest_main)

# System
add_executable(controlled_system ${CMAKE_CURRENT_SOURCE_DIR}/system/controlled.cc)
target_link_libraries(controlled_system damotion GTest::gtest_main)

add_executable(constraint ${CMAKE_CURRENT_SOURCE_DIR}/system/constraint.cc)
target_link_libraries(constraint damotion GTest::gtest_main)

# Common
add_executable(profiler ${CMAKE_CURRENT_SOURCE_DIR}/common/profiler.cc)
target_link_libraries(profiler damotion GTest::gtest_main)

# Control
add_executable(osc ${CMAKE_CURRENT_SOURCE_DIR}/control/osc.cc)
target_link_libraries(osc damotion GTest::gtest_main)

# Solvers
add_executable(program_test ${CMAKE_CURRENT_SOURCE_DIR}/solvers/program.cc)
target_link_libraries(program_test damotion GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(codegen_test)
gtest_discover_tests(eigen_wrapper_test)
gtest_discover_tests(pinocchio_wrapper_test)
gtest_discover_tests(controlled_system)
gtest_discover_tests(constraint)
gtest_discover_tests(log)
gtest_discover_tests(profiler)
gtest_discover_tests(osc)
gtest_discover_tests(program_test)