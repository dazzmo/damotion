cmake_minimum_required(VERSION 3.20)
project(damotion VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3")

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

IF(APPLE)
    # Fix linking on 10.14+. See https://stackoverflow.com/questions/54068035
    LINK_DIRECTORIES(/usr/local/lib)
ENDIF()

# Add damotion library
add_subdirectory(damotion)

option(BUILD_TESTS "Build damotion with tests" OFF)
if(BUILD_TESTS)
  add_subdirectory(test)
endif(BUILD_TESTS)

## We install the export that we defined above
## Export the targets to a script
## This will install the import script damotionTargets.cmake
## When findpackaged by other scripts, it will load the targets defined
## in the export damotion-export.
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/damotionConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Create damotionConfig.cmake with extra info from damotionConfig.cmake.in
# This file is necessary to find_package the library damotionConfig.
set(INSTALL_CONFIGDIR lib/cmake/damotion)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/damotionConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/damotionConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

# Install config and export files
install(EXPORT damotion-export
  FILE damotionTargets.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
  NAMESPACE damotion::
)

## Install the config and configversion
install(FILES
${CMAKE_CURRENT_BINARY_DIR}/damotionConfig.cmake
${CMAKE_CURRENT_BINARY_DIR}/damotionConfigVersion.cmake
# ${CMAKE_CURRENT_LIST_DIR}/cmake/FindGflags.cmake
# ${CMAKE_CURRENT_LIST_DIR}/cmake/FindGlog.cmake
DESTINATION ${INSTALL_CONFIGDIR}
)
