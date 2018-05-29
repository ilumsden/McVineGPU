#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <curand.h>
#include <curand_kernel.h>

__global__ void initArray(float* data, const int size, const float val);

__device__ void calculateTime(float* ts, float* pts, 
                              float x, float y, float z, float zdiff,
                              float va, float vb, float vc,
                              const float amin, const float amax,
                              const float bmin, const float bmax,
                              const int key, const int off1, int &off2);

__global__ void intersectBox(float* rx, float* ry, float* rz,
                             float* vx, float* vy, float* vz,
                             const float xmin, const float xmax,
                             const float ymin, const float ymax,
                             const float zmin, const float zmax,
                             const int N, float* ts, float* pts);

__global__ void simplifyTimes(const float* ts, const int N, const int groupSize, float* simp);

__global__ void prepRand(curandState *state, int seed);

__device__ void randCoord(float* inters, float* time, float *sx, float *sy, float *sz, curandState *state);

__global__ void calcScatteringSites(float* ts, float* int_pts, 
                                    float* pos, curandState *state, const int N);

#endif
