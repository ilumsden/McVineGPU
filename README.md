# McVineGPU_Test

This repo is a personal repo for performing proof-of-concept tests for ideas related to integrating GPU acceleration through CUDA to [McVine](https://github.com/mcvine/mcvine).

## Dependencies:

The following dependencies are required to run the code in this repository:
* [CMake 5.1.2 or newer](https://cmake.org/)
* [CUDA 8.0 or newer](https://developer.nvidia.com/cuda-toolkit)
* [A C++ compiler that supports C++11](https://cmake.org/://gcc.gnu.org/gcc-4.8/)

## Organization:

The code in this repository is organized as follows:
* `bash`: This directory stores bash scripts that can be used to easily compile and run the code.
* `cuda`: This directory stores the main CUDA code for the repository, such as
  * CUDA Kernels
  * CUDA Device Code
  * Host (CPU) Code that calls CUDA Kernels
* `src`: This directory stores any plain C++ (no CUDA) code that is needed. Currently, it only contains the main function for the non-test code.
* `test`: This directory stores all the C++ (regular and CUDA) code that is directly used for testing (i.e. unit tests).
