cmake_minimum_required(VERSION 3.5.1)

project(TestSuite_MVGPU)

option(USE_CUDA "Use CUDA" ON)

set(CMAKE_CUDA_COMPILER nvcc)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB test_sources "*.cpp")
message(STATUS "Sources are ${test_sources}")
file(GLOB test_cuda "*.cu")
file(GLOB test_lib "../cuda/*.cu")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda)

get_property(BUILD_TESTING DIRECTORY .. PROPERTY BUILD_TESTING)
get_property(USE_CUDA DIRECTORY .. PROPERTY USE_CUDA)

if(BUILD_TESTING)
    message(STATUS "Building tests.")
    add_executable(${PROJECT_NAME} ${test_sources})
    if(USE_CUDA)
        find_package(CUDA 8.0)
        if(NOT CUDA_FOUND)
            message(STATUS "CUDA not found.")
            set(USE_CUDA OFF)
        else()
            list(APPEND CUDA_NVCC_FLAGS "--relocatable-device-code=true")
            CUDA_ADD_LIBRARY(testLib ${test_cuda} ${test_lib} STATIC)
            target_link_libraries(${PROJECT_NAME} testLib)
        endif()
    endif()
endif(BUILD_TESTING)
