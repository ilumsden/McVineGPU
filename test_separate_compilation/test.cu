#include "test.hpp"

__global__ void testDrive()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d\n", index);
    //if (index == 0)
    //{
        int num = test();
        printf("num = %d\n", num);
    //}
}

void runTest()
{
    testDrive<<<1, 32>>>();
    cudaDeviceSynchronize();
}
