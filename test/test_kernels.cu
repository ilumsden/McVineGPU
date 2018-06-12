#include "test_kernels.hpp"
#include "Kernels.hpp"

__global__ void triangleTest(float *ts, float *pts)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0)
    {
        int offset = 0;
        intersectTriangle(ts, pts,
                          0, 0, 0,
                          0, 0, 1,
                          0, 1, 1,
                          1, -1, 1,
                          -1, -1, 1,
                          0, offset);
    }
}
