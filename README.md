# McVineGPU

This repo is for developing integration of GPU acceleration through CUDA for [McVine](https://github.com/mcvine/mcvine).

## Dependencies:

The following dependencies are required to run the code in this repository:
* [CMake 3.8 or newer](https://cmake.org/)
* [CUDA 8.0 or newer](https://developer.nvidia.com/cuda-toolkit)
  * This project supports separable compilation of device code. As a result, this should only be run on a Nvidia GPU with a Compute Capability of 2.0 or higher. You can find a list of GPUs and their Compute Capabilities [here](https://developer.nvidia.com/cuda-gpus). __Note:__ The hyperlinked website does _not_ list all CUDA capable Nvidia GPUs. Some GPUs that aren't listed (i.e. GeForce GTX 1050 Mobile) can still be used.
* [A C++ compiler that supports C++11](https://gcc.gnu.org/gcc-4.8/)

## Organization:

The code in this repository is organized as follows:
* `bash`: This directory stores bash scripts that can be used to easily compile and run the code.
* `cuda`: This directory stores the main CUDA code for the repository, such as
  * CUDA Kernels
  * CUDA Device Code
  * Host (CPU) Code that calls CUDA Kernels
* `src`: This directory stores any plain C++ (no CUDA) code that is needed. Currently, it only contains the main function for the non-test code.
* `test`: This directory stores the unit tests for the code base. All these tests are made with *googletest*.
* `test_separate_compilation`: This directory is not part of the main repository. It is an example directory that shows how to manually compile a CUDA project with separable compilation for device code.
