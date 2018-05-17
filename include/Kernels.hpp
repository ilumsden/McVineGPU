#include <curand.h>
#include <curand_kernel.h>

__device__ void calculate_time(float* ts, float x, float y, float z,
                               float va, float vb, float vc,
                               const float A, const float B, const int offset);

__global__ void intersectRectangle(float* rx, float* ry, float* rz,
                                   float* vx, float* vy, float* vz,
                                   const float X, const float Y, const float Z,
                                   const int N, float* ts);

__device__ void randCoord(float x0, float x1, float y0, float y1, float z0, float z1, float t0, float t1, float *sx, float *sy, float *sz, curandState *state);

__global__ void calcScatteringSites(const float* rx, const float* ry, const float* rz,
                                    const float* vx, const float* vy, const float* vz,
                                    const float X, const float Y, const float Z,
                                    const float* ts, float* pos, curandState *state, const int N);
