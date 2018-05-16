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

__device__ float randCoord(float x0, float x1, float t0, float t1, curandState *state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float dt = t1 - t0;
    float m = (x1 - x0)/dt;
    float randt = curand_uniform(&(state[index]));
    randt *= dt;
    //float randx = x0 + m * randt;
    return x0 + m*randt;
}

__global__ void calcScatteringSites(const float* rx, const float* ry, const float* rz,
                                    const float* vx, const float* vy, const float* vz,
                                    const float X, const float Y, const float Z,
                                    const float* ts, float* pos, curandState *state, const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float x0, y0, z0, x1, y1, z1;
    if (index < N)
    {
        float t0 = -5;
        float t1 = -5;
        int key0 = -1;
        int key1 = -1;
        curand_init(1337, index, 0, &(state[index]));
        __syncthreads();
        for (int i = 0; i < 6; i++)
        {
            if (ts[6*index + i] != -1)
            {
                if (t0 == -5)
                {
                    t0 = ts[6*index + i];
                    key0 = i;
                }
                else if (t1 == -5)
                {
                    t1 = ts[6*index + i];
                    key1 = i;
                }
                else
                {
                    // Some type of error handling
                    return;
                }
            }
        }
        __syncthreads();
        if (t0 != -5 && t1 != -5)
        {
            if (t0 > t1)
            {
                float tmpt;
                int tmpk;
                tmpt = t0;
                t0 = t1;
                t1 = tmpt;
                tmpk = key0;
                key0 = key1;
                key1 = tmpk;
            }
            switch(key0)
            {
                case 0: z0 = Z/2; x0 = rx[index] + vx[index]*t0; y0 = ry[index] + vy[index]*t0; break;
                case 1: z0 = -Z/2; x0 = rx[index] + vx[index]*t0; y0 = ry[index] + vy[index]*t0; break;
                case 2: x0 = X/2; y0 = ry[index] + vy[index]*t0; z0 = rz[index] + vz[index]*t0; break;
                case 3: x0 = -X/2; y0 = ry[index] + vy[index]*t0; z0 = rz[index] + vz[index]*t0; break;
                case 4: y0 = Y/2; x0 = rx[index] + vx[index]*t0; z0 = rz[index] + vz[index]*t0; break;
                case 5: y0 = -Y/2; x0 = rx[index] + vx[index]*t0; z0 = rz[index] + vz[index]*t0; break;
                default: return;// Some type of error handling
            }
            __syncthreads();
            switch(key1)
            {
                case 0: z1 = Z/2; x1 = rx[index] + vx[index]*t1; y1 = ry[index] + vy[index]*t1; break;
                case 1: z1 = -Z/2; x1 = rx[index] + vx[index]*t1; y1 = ry[index] + vy[index]*t1; break;
                case 2: x1 = X/2; y1 = ry[index] + vy[index]*t1; z1 = rz[index] + vz[index]*t1; break;
                case 3: x1 = -X/2; y1 = ry[index] + vy[index]*t1; z1 = rz[index] + vz[index]*t1; break;
                case 4: y1 = Y/2; x1 = rx[index] + vx[index]*t1; z1 = rz[index] + vz[index]*t1; break;
                case 5: y1 = -Y/2; x1 = rx[index] + vx[index]*t1; z1 = rz[index] + vz[index]*t1; break;
                default: return;// Some type of error handling
            }
            __syncthreads();
            pos[3*index + 0] = randCoord(x0, x1, t0, t1, state);
            pos[3*index + 1] = randCoord(y0, y1, t0, t1, state);
            pos[3*index + 2] = randCoord(z0, z1, t0, t1, state);
        }
    }
}
