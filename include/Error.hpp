#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H

#define CudaError( err ) __cudaErrorwithCode(err, __FILE__, __LINE__)
#define CudaErrorNoCode() __cudaErrorNoCode(__FILE__, __LINE__)
 
#include <cstdio>
#include <cstdlib>

inline void __cudaErrorwithCode(cudaError_t err, const char *file, const int line)
{
#ifdef CUDA_ERROR_H
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CudaError() failed at %s:%i\nError Message: %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif
    return;
}

inline void __cudaErrorNoCode(const char *file, const int line)
{
#ifdef CUDA_ERROR_H
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CudaError() failed at %s:%i\nError Message: %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
    // This code affects performance. Only uncomment during debugging or if needed.
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CudaError() with sync failed at %s:%i\nError Message: %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif
    return;
}

#endif
