#ifndef UTIL_KERNELS_HPP
#define UTIL_KERNELS_HPP

/* This file lists the function declarations for all CUDA kernels.
 * It will likely be broken into several smaller, more specific
 * files later.
 */

#include <curand.h>
#include <curand_kernel.h>

#include "Vec3.hpp"

#ifndef PI
#define PI 3.14159265358979323846f
#endif

/* This function initializes the contents of the data array with the
 * value val.
 * This function can be called from host.
 */
template <typename T>
__global__ void initArray(T* data, const int size, const T val)
{
    /* This is done simply to allow the host compiler (g++, clang, etc.)
     * to successfully compile the driver cpp file. When running,
     * only the code in the __CUDA_ARCH__ block will be used.
     */
#if defined(__CUDA_ARCH__)
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
#else
    int idx = 0;
    int stride = 0;
#endif
    for (int i = idx; i < size; i += stride)
    {
        data[i] = val;
    }
}

/* This function solves the quadratic equation given values a, b, and c.
 * The results of the equation are stored in x0 and x1.
 * This function can be called on device only.
 */
__device__ bool solveQuadratic(float a, float b, float c, 
                               float &x0, float &x1);

/* This function takes the times produced by the intersect functions above
 * for solids (i.e. Box, Sphere, Cylinder, etc.) and reduces the array so
 * that there are only 2 times per neutron. If there are no meaningful
 * times for a neturon, the times are simplified to 2 -1s. The simplified
 * data is stored in simp. N is the number of neutrons, and groupSize is
 * the number of times per neutron in ts.
 * This function can be called from host.
 */
__global__ void simplifyTimes(const float* ts, const int N, 
                              const int groupSize, float* simp);

/* This function seeds and initializes a cuRand random number generator
 * using the cuRand States stored in state and the seed value "seed."
 * This function can be called from host.
 */
__global__ void prepRand(curandState *state, int seed);

#endif
