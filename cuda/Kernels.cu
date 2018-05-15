#include "Kernels.hpp"

/* This is a device-only helper function for determining the time
 * it takes a ray to intersect the rectangle specified by the `intersectRectangle`
 * function.
 * It is a CUDA version of the intersectRectangle function from ArrowIntersector.cc
 * in McVine (mcvine/packages/mccomposite/lib/geometry/visitors/ArrowIntersector.cc).
 */
__device__ void calculate_time(float* ts, float x, float y, float z,
                               float va, float vb, float vc, const float A, const float B, const int offset)
{
    __syncthreads();
    float t = (0-z)/vc;
    float r1x = x+va*t; 
    float r1y = y+vb*t;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (fabs(r1x) < A/2 && fabs(r1y) < B/2)
    {
        ts[offset + index*6] = t;
    }
    else
    {
        ts[offset + index*6] = -1;
    }
}

/* This is a global CUDA function for controlling the calculation of intersection
 * times. It is a CUDA version of the visit function from ArrowIntersector.cc in
 * McVine (mcvine/packages/mccomposite/lib/geometry/visitors/ArrowIntersector.cc).
 */
__global__ void intersectRectangle(
    float* rx, float* ry, float* rz,
    float* vx, float* vy, float* vz,
    const float X, const float Y, const float Z, const int N,
    float* ts)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        if (vz[index] != 0)
        {
            calculate_time(ts, rx[index], ry[index], rz[index]-Z/2, vx[index], vy[index], vz[index], X, Y, 0);
            calculate_time(ts, rx[index], ry[index], rz[index]+Z/2, vx[index], vy[index], vz[index], X, Y, 1);
        }
        else
        {
            ts[index*6] = -1;
            ts[index*6 + 1] = -1;
        }
        if (vx[index] != 0)
        {
            calculate_time(ts, ry[index], rz[index], rx[index]-X/2, vy[index], vz[index], vx[index], Y, Z, 2);
            calculate_time(ts, ry[index], rz[index], rx[index]+X/2, vy[index], vz[index], vx[index], Y, Z, 3);
        }
        else
        {
            ts[index*6 + 2] = -1;
            ts[index*6 + 3] = -1;
        }
        if (vy[index] != 0)
        {
            calculate_time(ts, rz[index], rx[index], ry[index]-Y/2, vz[index], vx[index], vy[index], Z, X, 4);
            calculate_time(ts, rz[index], rx[index], ry[index]+Y/2, vz[index], vx[index], vy[index], Z, X, 5);
        }
        else
        {
            ts[index*6 + 4] = -1;
            ts[index*6 + 5] = -1;
        }
    }
}
/*
__device__ float randCoord(float x1, float y1, float z1, float x2, float y2, float z2, curandState *state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1337, index, 0, &state[index]);
    
}

__global__ void calcScatteringSites(const float* rx, const float* ry, const float* rz,
                                    const float* vx, const float* vy, const float* vz,
                                    const float* ts, float* pos, curandState *state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
}*/
