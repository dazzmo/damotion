find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
link_directories(${GTEST_LIB_DIRS})

enable_testing()

add_executable(codegen_test ${CMAKE_CURRENT_SOURCE_DIR}/utils/codegen.cc)
target_link_libraries(codegen_test casadi_project GTest::gtest_main)

add_executable(eigen_wrapper_test ${CMAKE_CURRENT_SOURCE_DIR}/utils/eigen_wrapper.cc)
target_link_libraries(eigen_wrapper_test casadi_project GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(codegen_test)
gtest_discover_tests(eigen_wrapper_test)