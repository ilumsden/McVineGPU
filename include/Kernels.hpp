#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <curand.h>
#include <curand_kernel.h>

__global__ void initArray(float* data, const int size, const float val);

__device__ void intersectRectangle(float* ts, float* pts, 
                                   float x, float y, float z, float zdiff,
                                   float va, float vb, float vc,
                                   const float A, const float B,
                                   const int key, const int groupSize,
                                   const int off1, int &off2);

__device__ void intersectCylinderSide(float *ts, float *pts,
                                      float x, float y, float z,
                                      float vx, float vy, float vz,
                                      const float r, const float h,
                                      int &offset);

__device__ void intersectCylinderTopBottom(float *ts, float *pts,
                                           float x, float y, float z,
                                           float vx, float vy, float vz,
                                           const float r, const float h,
                                           int &offset);

__device__ void intersectTriangle(float *ts, float *pts,
                                  const float x, const float y, const float z,
                                  const float vx, const float vy, const float vz,
                                  const float aX, const float aY, const float aZ,
                                  const float bX, const float bY, const float bZ,
                                  const float cX, const float cY, const float cZ,
                                  const int off1, int &off2);

__global__ void intersectBox(float* rx, float* ry, float* rz,
                             float* vx, float* vy, float* vz,
                             const float X, const float Y, const float Z,
                             const int N, float* ts, float* pts);

__global__ void intersectCylinder(float *rx, float *ry, float *rz,
                                  float *vx, float *vy, float *vz,
                                  const float r, const float h,
                                  const int N, float *ts, float *pts);

__global__ void intersectPyramid(float *rx, float *ry, float *rz,
                                 float *vx, float *vy, float *vz,
                                 const float X, const float Y, const float H,
                                 const int N, float *ts, float *pts);

__global__ void simplifyTimes(const float* ts, const int N, const int groupSize, float* simp);

__global__ void prepRand(curandState *state, int seed);

__device__ void randCoord(float* inters, float* time, float *sx, float *sy, float *sz, curandState *state);

__global__ void calcScatteringSites(float* ts, float* int_pts, 
                                    float* pos, curandState *state, const int N);

#endif
