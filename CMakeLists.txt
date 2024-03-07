cmake_minimum_required(VERSION 3.20)
project(casadi_project VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/)  

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

find_package(Eigen3 3.3 REQUIRED)
find_package(glog REQUIRED)
find_package(casadi REQUIRED)
find_package(pinocchio REQUIRED)
find_package(qpOASES REQUIRED)

# Collect source files
file (GLOB_RECURSE SOURCE_FILES 
"utils/*.cc"
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC 
Eigen3::Eigen 
pinocchio::pinocchio 
casadi 
glog 
ipopt 
qpOASES
)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(${PROJECT_NAME} 
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>"
)

target_compile_definitions(${PROJECT_NAME} PRIVATE "-DINCLUDE_DIR=\"${PROJECT_SOURCE_DIR}\"")

add_subdirectory(test)

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