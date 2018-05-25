#include <cstdio>

#include "Kernels.hpp"

__global__ void initArray(float *data, int size, const float val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride)
    {
        data[i] = val;
    }
}

/* This is a device-only helper function for determining the time
 * it takes a ray to intersect the rectangle specified by the `intersectRectangle`
 * function.
 * It is a CUDA version of the intersectRectangle function from ArrowIntersector.cc
 * in McVine (mcvine/packages/mccomposite/lib/geometry/visitors/ArrowIntersector.cc).
 */
__device__ void calculate_time(float* ts, float* pts,
                               float x, float y, float z, float zdiff,
                               float va, float vb, float vc, 
                               const float A, const float B, 
                               const int key, const int off1, int &off2)
{
    z += zdiff;
    float t = (0-z)/vc;
    float r1x = x+va*t; 
    float r1y = y+vb*t;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (fabs(r1x) < A/2 && fabs(r1y) < B/2)
    {
        float ix, iy, iz;
        if (key == 0)
        {
            ix = r1x;
            iy = r1y;
            iz = -zdiff;
        }
        else if (key == 1)
        {
            iy = r1x;
            iz = r1y;
            ix = -zdiff;
        }
        else
        {
            iz = r1x;
            ix = r1y;
            iy = -zdiff;
        }
        if (off2 == 0 || off2 == 3)
        {
            pts[6*index + off2] = ix;
            pts[6*index + off2 + 1] = iy;
            pts[6*index + off2 + 2] = iz;
            off2 += 3;
        }
        ts[off1 + index*6] = t;
    }
    else
    {
        ts[off1 + index*6] = -1;
    }
}

/* This is a global CUDA function for controlling the calculation of intersection
 * times. It is a CUDA version of the visit function from ArrowIntersector.cc in
 * McVine (mcvine/packages/mccomposite/lib/geometry/visitors/ArrowIntersector.cc).
 */
__global__ void intersectBlock(
    float* rx, float* ry, float* rz,
    float* vx, float* vy, float* vz,
    const float X, const float Y, const float Z, const int N,
    float* ts, float* pts)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        int offset = 0;
        if (vz[index] != 0)
        {
            calculate_time(ts, pts, rx[index], ry[index], rz[index], -Z/2, vx[index], vy[index], vz[index], X, Y, 0, 0, offset);
            calculate_time(ts, pts, rx[index], ry[index], rz[index], Z/2, vx[index], vy[index], vz[index], X, Y, 0, 1, offset);
        }
        else
        {
            ts[index*6] = -1;
            ts[index*6 + 1] = -1;
        }
        if (vx[index] != 0)
        {
            calculate_time(ts, pts, ry[index], rz[index], rx[index], -X/2, vy[index], vz[index], vx[index], Y, Z, 1, 2, offset);
            calculate_time(ts, pts, ry[index], rz[index], rx[index], X/2, vy[index], vz[index], vx[index], Y, Z, 1, 3, offset);
        }
        else
        {
            ts[index*6 + 2] = -1;
            ts[index*6 + 3] = -1;
        }
        if (vy[index] != 0)
        {
            calculate_time(ts, pts, rz[index], rx[index], ry[index], -Y/2, vz[index], vx[index], vy[index], Z, X, 2, 4, offset);
            calculate_time(ts, pts, rz[index], rx[index], ry[index], Y/2, vz[index], vx[index], vy[index], Z, X, 2, 5, offset);
        }
        else
        {
            ts[index*6 + 4] = -1;
            ts[index*6 + 5] = -1;
        }
    }
}

__global__ void simplifyTimes(const float *times, const int N, const int groupSize, float *simp)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        int count = 0;
        for (int i = 0; i < groupSize; i++)
        {
            if (times[groupSize * index + i] != -1 && count < 2)
            {
                simp[2*index+count] = times[groupSize*index+i];
                count++;
            }
        }
    }
}

__global__ void prepRand(curandState *state, int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(((seed << 10) + idx), 0, 0, &state[idx]); 
}

__device__ void randCoord(float* inters, float* time , float *sx, float *sy, float *sz, curandState *state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float dt = time[1] - time[0];
    float mx = (inters[3] - inters[0])/dt;
    float my = (inters[4] - inters[1])/dt;
    float mz = (inters[5] - inters[2])/dt;
    float randt = curand_uniform(&(state[index]));
    randt *= dt;
    *sx = inters[0] + mx*randt;
    *sy = inters[1] + my*randt;
    *sz = inters[2] + mz*randt;
}

__global__ void calcScatteringSites(const float X, const float Y, const float Z,
                                    float* ts, float* int_pts, float* pos, curandState *state, const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        if (ts[2*index] != -5 && ts[2*index+1] != -5)
        {
            if (ts[2*index] > ts[2*index+1])
            {
                float tmpt, tmpc;
                tmpt = ts[2*index];
                ts[2*index] = ts[2*index+1];
                ts[2*index+1] = tmpt;
                for (int i = 6*index; i < 6*index+3; i++)
                {
                    tmpc = int_pts[i];
                    int_pts[i] = int_pts[i + 3];
                    int_pts[i + 3] = tmpc;
                }
            }
            randCoord(&(int_pts[6*index]), &(ts[2*index]), &(pos[3*index + 0]), &(pos[3*index + 1]), &(pos[3*index + 2]), state);
        }
        else
        {
            pos[3*index + 0] = 20 * X;
            pos[3*index + 1] = 20 * Y;
            pos[3*index + 2] = 20 * Z;
        }
    }
}
