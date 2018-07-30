#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H

/* This file defines 2 functions for error checking CUDA
 * code (primarilty CUDA API calls).
 * The contents of this file were simply found on the Internet and
 * copied here. It is the standard way of error checking your CUDA
 * code.
 */

#include <curand.h>

/* This function macro is used to error check any CUDA function
 * or process that directly returns an error code. It is primarily
 * meant to be used with CUDA API functions. It should contain whatever
 * function that is being checked for errors. For example:
 *     CudaErrchk( cudaMalloc(...) );
 */
#define CudaErrchk( err ) __cudaErrorwithCode(err, __FILE__, __LINE__)
/* This function macro is used to error check any CUDA function, process,
 * or kernel that does not directly return an error code. It is primarily
 * meant to be used to error check global kernels and CUDA code that
 * is not part of the CUDA API. For example:
 *     kernel<<<numBlocks, blockSize>>>(...);
 *     CudaErrchkNoCode();
 */
#define CudaErrchkNoCode() __cudaErrorNoCode(__FILE__, __LINE__)

#define CuRandErrchk( err ) __cuRandErrorwithCode(err, __FILE__, __LINE__)
 
#include <cstdio>
#include <cstdlib>
#include <string>

/* This function checks if the CUDA error code passed as the
 * first parameter represents a sucessful operation or not.
 * If not, it will print a meaningful error message to stderr.
 */
inline void __cudaErrorwithCode(cudaError_t err, const char *file, const int line)
{
#ifdef CUDA_ERROR_H
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CudaErrchk() failed at %s:%i\nError Message: %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif
    return;
}

/* This function checks if the last accessible CUDA error code 
 * represents a sucessful operation or not.
 * If not, it will print a meaningful error message to stderr.
 * The second half of the function will pause all CPU instructions until
 * all currently running GPU code is finished. It will then get any CUDA
 * errors associated with the synchronization and print a meaningful
 * error message if the synchronization was not sucessful.
 * This second test does affect performance, so it should only be used
 * during debugging, testing, or if needed. Otherwise, it should be
 * commented out. 
 */
inline void __cudaErrorNoCode(const char *file, const int line)
{
#ifdef CUDA_ERROR_H
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CudaErrchk() failed at %s:%i\nError Message: %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
    // This code affects performance. Only uncomment during debugging or if needed.
    /*err = cudaDeviceSynchronize();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CudaErrchk() with sync failed at %s:%i\nError Message: %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }*/
#endif
    return;
}

inline void __cuRandErrorwithCode(curandStatus_t err, const char *file, const int line)
{
    if (err != CURAND_STATUS_SUCCESS)
    {
        std::string errmsg;
        switch(err)
        {
            case CURAND_STATUS_VERSION_MISMATCH:
                errmsg = "CURAND_STATUS_VERSION_MISMATCH"; break;
            case CURAND_STATUS_NOT_INITIALIZED:
                errmsg = "CURAND_STATUS_NOT_INITIALIZED"; break;
            case CURAND_STATUS_ALLOCATION_FAILED:
                errmsg = "CURAND_STATUS_ALLOCATION_FAILED"; break;
            case CURAND_STATUS_TYPE_ERROR:
                errmsg = "CURAND_STATUS_TYPE_ERROR"; break;
            case CURAND_STATUS_OUT_OF_RANGE:
                errmsg = "CURAND_STATUS_OUT_OF_RANGE"; break;
            case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
                errmsg = "CURAND_STATUS_LENGTH_NOT_MULTIPLE"; break;
            case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                errmsg = "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED"; break;
            case CURAND_STATUS_LAUNCH_FAILURE:
                errmsg = "CURAND_STATUS_LAUNCH_FAILURE"; break;
            case CURAND_STATUS_PREEXISTING_FAILURE:
                errmsg = "CURAND_STATUS_PREEXISTING_FAILURE"; break;
            case CURAND_STATUS_INITIALIZATION_FAILED:
                errmsg = "CURAND_STATUS_INITIALIZATION_FAILED"; break;
            case CURAND_STATUS_ARCH_MISMATCH:
                errmsg = "CURAND_STATUS_ARCH_MISMATCH"; break;
            case CURAND_STATUS_INTERNAL_ERROR:
                errmsg = "CURAND_STATUS_INTERNAL_ERROR"; break;
            default:
                errmsg = "CURAND_STATUS_UNKNOWN"; break;
        }
        fprintf(stderr, "CuRandErrchk failed at %s:%i\nError Message: %s\n", file, line, errmsg.c_str());
        exit(-1);
    }
}

#endif
