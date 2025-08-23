cmake_minimum_required(VERSION 2.8.3)
project(my3d)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS " -std=c++17 -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -march=native")

find_package(OpenCV 4 REQUIRED)
if (OpenCV_FOUND)
    message("OpenCV found.")
    message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
endif()

find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    message("EIGEN3 found.")
    message("EIGEN3_INCLUDE_DIRS: ${EIGEN3_INCLUDE_DIRS}")
endif()

find_package(Boost 1.70 REQUIRED 
    COMPONENTS 
    unit_test_framework 
    test_exec_monitor
    filesystem 
    # archive 
    serialization)
if (Boost_FOUND) 
    message("Boost ${Boost_VERSION} found.")
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/3rd_party/GraphOpt)

set(GOPT_INCLUDE_DIRS 
    "${CMAKE_SOURCE_DIR}/3rd_party/GraphOpt"
    "${CMAKE_SOURCE_DIR}/3rd_party/GraphOpt/gopt")
set(GOPT_LIBRARIES gopt_graph gopt_loss gopt_solver gopt_util)

add_subdirectory(${CMAKE_SOURCE_DIR}/3rd_party/SimpleJSON)
set(SimpleJSON_INCLUDE_DIRS 
    "${CMAKE_SOURCE_DIR}/3rd_party/SimpleJSON"
    "${CMAKE_SOURCE_DIR}/3rd_party/SimpleJSON/simple_json")
set(SimpleJSON_LIBRARIES simple_json)


add_subdirectory(${CMAKE_SOURCE_DIR}/3rd_party/TinyEXIF)
set(TinyEXIF_INCLUDE_DIRS 
    "${CMAKE_SOURCE_DIR}/3rd_party"
    "${CMAKE_SOURCE_DIR}/3rd_party/TinyEXIF")

include_directories(
    ${CMAKE_SOURCE_DIR}/my3d 
    ${GOPT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${TinyEXIF_INCLUDE_DIRS}
    ${SimpleJSON_INCLUDE_DIRS}
)

add_subdirectory(my3d/utils)
add_subdirectory(my3d/estimators)
add_subdirectory(my3d/base)
add_subdirectory(my3d/feature)
add_subdirectory(my3d/sfm)
add_subdirectory(my3d/applications)
