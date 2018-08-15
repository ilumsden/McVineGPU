# Testing Suite

All the files in this directory (and its subdirectories) are for unit testing the code in the rest of the repository. The suite is organized as follows:
* `run_tests.cpp`: This file is the starting point for running the test suite. The `test` executable will use this file for the `int main()` function.
* `gtest`: This directory contains the files for the actual *googletest*-based tests.
* `include`: This directory contains any header files needed during testing. This will include definitions of CUDA kernels and other C++ functions that are required to separate the CUDA code from *googletest*. If mock classes (based on *googlemock*) are added, they will also be found here.
* `src`: This directory contains the implementations of the functions defined in the files from the `include` directory.
