find_package(GTest REQUIRED)
find_package(gflags REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
link_directories(${GTEST_LIB_DIRS})

enable_testing()

# Create list of files

SET(TEST_FILES
common/function.cc
utils/codegen.cc
utils/eigen_wrapper.cc
utils/pinocchio_model.cc
utils/log.cc
utils/expression.cc
system/controlled.cc
common/profiler.cc
control/osc.cc
solvers/cost.cc
solvers/program.cc
solvers/ipopt.cc
)

include(GoogleTest)

foreach(current_file IN LISTS TEST_FILES)
get_filename_component(current_name ${current_file} NAME_WE)
    add_executable("test_${current_name}" ${current_file})
    target_link_libraries("test_${current_name}" damotion GTest::gtest_main glog gflags)
endforeach()
