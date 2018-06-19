# McVineGPU_Test

This repo is a personal repo for performing proof-of-concept tests for ideas related to integrating GPU acceleration through CUDA to [McVine](https://github.com/mcvine/mcvine).

## Dependencies:

The following dependencies are required to run the code in this repository:
* [CMake 3.8 or newer](https://cmake.org/)
* [CUDA 8.0 or newer](https://developer.nvidia.com/cuda-toolkit)
  * This project will support separable compilation of device code in the future. As a result, this should only be run on a Nvidia GPU with a Compute Capability of 2.0 or higher. You can find a list of GPUs and their Compute Capabilities [here](https://developer.nvidia.com/cuda-gpus).
* [A C++ compiler that supports C++11](https://gcc.gnu.org/gcc-4.8/)

## Organization:

The code in this repository is organized as follows:
* `bash`: This directory stores bash scripts that can be used to easily compile and run the code.
* `cuda`: This directory stores the main CUDA code for the repository, such as
  * CUDA Kernels
  * CUDA Device Code
  * Host (CPU) Code that calls CUDA Kernels
* `src`: This directory stores any plain C++ (no CUDA) code that is needed. Currently, it only contains the main function for the non-test code.
* `test`: This directory stores all the C++ (regular and CUDA) code that is directly used for testing (i.e. unit tests).
* `test_separate_compilation`: This directory is not part of the main repository. It is an example directory that shows how to manually compile a CUDA project with separable compilation for device code.
