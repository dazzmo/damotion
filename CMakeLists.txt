cmake_minimum_required(VERSION 3.20)
project(damotion VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -Wall")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/)  

option(WITH_QPOASES "Includes functionality for qpOASES" OFF)
option(WITH_IPOPT "Includes functionality for IPOPT" OFF)

add_subdirectory(damotion)

option(BUILD_TEST "Compile all tests" OFF)
if(BUILD_TESTS)
  add_subdirectory(test)
endif(BUILD_TESTS)