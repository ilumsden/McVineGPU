cmake_minimum_required(VERSION 2.8)

project(McVineGPUTest LANGUAGES CXX)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CXX_STANDARD 11)
# Other libraries can be added later if needed
set(CLIBS ${CUDA_LIBRARIES}) 
set(LIBS ${LIBS} ${CLIBS})
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION OFF)

list(APPEND CUDA_NVCC_FLAGS -g; -G)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(cuda)

file(GLOB_RECURSE CUDA_SRC cuda/*.cu)
file(GLOB_RECURSE SRC src/*.cpp)

cuda_add_library(cudaMcVine ${CUDA_SRC} STATIC)
set(LIBS ${LIBS} ${cudaMcVine})

message("Libs:")
message(${LIBS})

add_executable(McVineGPUTest ${SRC})
target_link_libraries(${PROG_NAME} ${LIBS})
