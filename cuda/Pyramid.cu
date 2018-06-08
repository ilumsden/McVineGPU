#include <cfloat>

#include "Error.hpp"
#include "Pyramid.hpp"
#include "Kernels.hpp"

void Pyramid::intersect(float *d_rx, float *d_ry, float *d_rz,
                        float *d_vx, float *d_vy, float *d_vz,
                        const int N, const int blockSize, const int numBlocks,
                        std::vector<float> &int_times, std::vector<float> &int_coords)
{
    float *device_time;
    CudaErrchk( cudaMalloc(&device_time, 5*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(device_time, 5*N, -5);
    CudaErrchkNoCode();
    float *intersect;
    CudaErrchk( cudaMalloc(&intersect, 6*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(intersect, 6*N, FLT_MAX);
    CudaErrchkNoCode();
    float *simp_times;
    CudaErrchk( cudaMalloc(&simp_times, 2*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(simp_times, 2*N, -5);
    CudaErrchkNoCode();
    int_times.resize(2*N);
    int_coords.resize(6*N);
    intersectPyramid<<<numBlocks, blockSize>>>(d_rx, d_ry, d_rz,
                                               d_vx, d_vy, d_vz,
                                               edgeX, edgeY, height,
                                               N, device_time, intersect);
    //CudaDeviceSynchronize();
    //printf("\n\nEnd Kernel.\n");
    /*std::vector<float> tmp;
    tmp.resize(5*N);
    CudaErrchk( cudaMemcpy(tmp.data(), device_time, 5*N*sizeof(float), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < (int)(tmp.size()); i++)
    {
        if (i % 5 == 0)
        {
            printf("Ray Index %i:\n", i/5);
        }
        printf("    Offset = %i: Time = %f\n", (i%5), tmp[i]);
    }*/
    cudaDeviceSynchronize();
    simplifyTimes<<<numBlocks, blockSize>>>(device_time, N, 5, simp_times);
    CudaErrchkNoCode();
    float *it = int_times.data();
    float *ic = int_coords.data();
    CudaErrchk( cudaMemcpy(it, simp_times, 2*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(ic, intersect, 6*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaFree(device_time) );
    CudaErrchk( cudaFree(intersect) );
    CudaErrchk( cudaFree(simp_times) );
}
