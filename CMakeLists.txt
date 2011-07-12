cmake_minimum_required(VERSION 2.8)

PROJECT(KMeans)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
SET(CMAKE_CXX_FLAGS "-O2 -g -Wall -pedantic")

FIND_PACKAGE(Eigen REQUIRED)

SET(KMEANS_SRC src/kmeans.cpp src/tools.cpp)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

INCLUDE_DIRECTORIES(include)

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

# ----- Testing -----

add_executable(test_xu test/test_real_data.cpp ${KMEANS_SRC})
target_link_libraries(test_xu m)

add_executable(test_lspi test/test_lspi.cpp ${KMEANS_SRC})
target_link_libraries(test_lspi m)
