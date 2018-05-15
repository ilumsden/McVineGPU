#include <curand.h>
#include <curand_kernel.h>

__device__ void calculate_time(float* ts, float x, float y, float z,
                               float va, float vb, float vc,
                               const float A, const float B, const int offset);

__global__ void intersectRectangle(float* rx, float* ry, float* rz,
                                   float* vx, float* vy, float* vz,
                                   const float X, const float Y, const float Z,
                                   const int N, float* ts);

/*__device__ float randCoord(float x1, float y1, float z1,
                           float x2, float y2, float z2, curandState *state);

__global__ void calcScatteringSites(const float* rx, const float* ry, const float* rz,
                                    const float* vx, const float* vy, const float* vz,
                                    const float* ts, float* pos, curandState *state);*/
