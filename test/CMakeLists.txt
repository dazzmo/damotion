find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
link_directories(${GTEST_LIB_DIRS})

# add local include
include_directories(${CMAKE_CURRENT_SOURCE_DIR} )

enable_testing()

add_executable(codegen_test ${CMAKE_CURRENT_SOURCE_DIR}/utils/codegen.cc)
target_link_libraries(codegen_test GTest::gtest_main casadi_project)

include(GoogleTest)
gtest_discover_tests(codegen_test)