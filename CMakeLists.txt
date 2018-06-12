# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 23 May 2018

cmake_minimum_required(VERSION 3.5)
project(bayes-filter-cpp)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

set(BFCPP_LIBRARY bfcpp)

add_subdirectory(dep)

include_directories(
    src
    ${EIGEN3_INCLUDE_DIR}
    ${CATCH2_INCLUDE_DIR}
)

file(GLOB SRC src/bayes_filter/*.cpp)

add_library(${BFCPP_LIBRARY} ${SRC})

install(TARGETS ${BFCPP_LIBRARY}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY src/bayes_cpp
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")

# Compile tests
file(GLOB TEST_SRC test/*.cpp)
add_executable(run_tests ${TEST_SRC})
target_link_libraries(run_tests ${BFCPP_LIBRARY})
