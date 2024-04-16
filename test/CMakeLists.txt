find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
link_directories(${GTEST_LIB_DIRS})

enable_testing()

# Create list of files

SET(TEST_FILES
${CMAKE_CURRENT_SOURCE_DIR}/common/function.cc 
${CMAKE_CURRENT_SOURCE_DIR}/utils/codegen.cc 
${CMAKE_CURRENT_SOURCE_DIR}/utils/eigen_wrapper.cc
${CMAKE_CURRENT_SOURCE_DIR}/utils/pinocchio_model.cc
${CMAKE_CURRENT_SOURCE_DIR}/utils/log.cc
${CMAKE_CURRENT_SOURCE_DIR}/utils/casadi.cc
${CMAKE_CURRENT_SOURCE_DIR}/utils/expression.cc
${CMAKE_CURRENT_SOURCE_DIR}/system/controlled.cc
${CMAKE_CURRENT_SOURCE_DIR}/common/profiler.cc
${CMAKE_CURRENT_SOURCE_DIR}/control/osc.cc
${CMAKE_CURRENT_SOURCE_DIR}/solvers/cost.cc
${CMAKE_CURRENT_SOURCE_DIR}/solvers/program.cc
)

include(GoogleTest)

foreach(current_file IN LISTS TEST_FILES)
get_filename_component(current_name ${current_file} NAME_WE)
    add_executable("test_${current_name}" ${current_file})
    target_link_libraries("test_${current_name}" damotion GTest::gtest_main)
    gtest_discover_tests("test_${current_name}")
endforeach()