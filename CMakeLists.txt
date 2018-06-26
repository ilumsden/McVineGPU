cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(McVineGPU LANGUAGES CXX CUDA)

option(BUILD_TESTING "Build tests." OFF)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/cuda/*.cu")
file(GLOB gpu_headers "${CMAKE_SOURCE_DIR}/include/*.hpp")
if(BUILD_TESTING)
    file(GLOB test_cuda_src "${CMAKE_SOURCE_DIR}/test/*.cu")
    file(GLOB test_cuda_headers "${CMAKE_SOURCE_DIR}/test/*.hpp")
    file(GLOB test_drivers "${CMAKE_SOURCE_DIR}/test/*.cpp")
endif()

message(STATUS "CUDA include directories are ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

list(APPEND CMAKE_CUDA_FLAGS "-cudart static")

add_library(cuMcVineLib STATIC ${gpu_source_files} ${gpu_headers})

target_compile_features(cuMcVineLib PUBLIC cxx_std_11)
set_target_properties(cuMcVineLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuMcVineLib curand)

add_executable(${PROJECT_NAME} ${cpu_source_files})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
target_link_libraries(${PROJECT_NAME} PRIVATE cuMcVineLib)

if(BUILD_TESTING)
    message(STATUS "Building test suite.")
    add_library(testMcVineLib ${test_cuda_src} ${test_cuda_headers})
    set_target_properties(testMcVineLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_features(testMcVineLib PUBLIC cxx_std_11)
    target_link_libraries(testMcVineLib cuMcVineLib)

    add_executable(Test_${PROJECT_NAME} ${test_drivers})
    target_compile_features(Test_${PROJECT_NAME} PUBLIC cxx_std_11)
    target_link_libraries(Test_${PROJECT_NAME} PRIVATE testMcVineLib)
endif()
