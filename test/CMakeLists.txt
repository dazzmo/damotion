find_package(GTest REQUIRED)
find_package(gflags REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
link_directories(${GTEST_LIB_DIRS})

enable_testing()

# Create list of files
SET(TEST_FILES
core/function.cc
casadi/eigen.cc
casadi/function.cc
optimisation/binding.cc
optimisation/objective_function.cc
optimisation/constraint_vector.cc
solvers/sparse.cc
solvers/qpoases.cc
solvers/ipopt.cc
symbolic/variable.cc
)

include(GoogleTest)

foreach(current_file IN LISTS TEST_FILES)
    get_filename_component(current_folder ${current_file} DIRECTORY)
    get_filename_component(current_name ${current_file} NAME_WE)
    add_executable("${current_folder}_${current_name}_unit_test" ${current_file})
    target_link_libraries("${current_folder}_${current_name}_unit_test" damotion GTest::gtest_main glog gflags)
endforeach()
