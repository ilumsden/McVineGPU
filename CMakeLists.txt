cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(McVineGPU)

option(USE_CUDA "Use CUDA" ON)

set(CMAKE_CUDA_COMPILER nvcc)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

option(BUILD_TESTING "Build tests." OFF)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/cuda/*.cu")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda)

#if( ("${CMAKE_VERSION}" VERSION_EQUAL 3.8) OR
#        ("${CMAKE_VERSION}" VERSION_GREATER 3.8) )
    # Modern CMake
#    if(USE_CUDA)
#        enable_language("CUDA")
#        add_executable(McVineGPUTest ${cpu_source_files} ${gpu_source_files})
#        set_property(TARGET McVineGPUTest PROPERTY CUDA_STANDARD 11)
#    else(USE_CUDA)
#        add_executable(McVineGPUTest ${cpu_source_files})
#    endif()
#else()
    # Old CMake
    add_executable(${PROJECT_NAME} ${cpu_source_files})
    if(USE_CUDA)
        find_package(CUDA 8.0)
        if(NOT CUDA_FOUND)
            message(STATUS "CUDA not found")
            set(USE_CUDA OFF)
        else()
            list(APPEND CUDA_NVCC_FLAGS "--std=c++11 -g -G -lcurand")
            CUDA_ADD_LIBRARY(McVineGPUTest_gpu ${gpu_source_files} STATIC)
            target_link_libraries(${PROJECT_NAME} McVineGPUTest_gpu)
        endif()
    endif()
#endif()

add_subdirectory(test)

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
