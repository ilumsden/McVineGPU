#include <cassert>
#include <cstdio>

#include "Vec3.hpp"

__host__ __device__ bool assert_almosteq(float a, float b);

__global__ void triangleTest(float *ts, float *pts);

__global__ void vec3Test(Vec3<float> *vectors, bool *res);
