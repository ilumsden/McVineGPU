#include <cfloat>

#include <chrono>

#include "Error.hpp"
#include "Kernels.hpp"
#include "Sphere.hpp"

void Sphere::intersect(float *d_rx, float *d_ry, float *d_rz,
                       float *d_vx, float *d_vy, float *d_vz,
                       const int N, const int blockSize, const int numBlocks,
                       std::vector<float> &int_times, std::vector<float> &int_coords)
{
    float *device_time;
    CudaErrchk( cudaMalloc(&device_time, 2*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(device_time, 2*N, -5);
    CudaErrchkNoCode();
    float *intersect;
    CudaErrchk( cudaMalloc(&intersect, 6*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(intersect, 6*N, FLT_MAX);
    CudaErrchkNoCode();
    int_times.resize(2*N);
    int_coords.resize(6*N);
    auto start = std::chrono::steady_clock::now();
    intersectSphere<<<numBlocks, blockSize>>>(d_rx, d_ry, d_rz,
                                              d_vx, d_vy, d_vz,
                                              radius,
                                              N, device_time, intersect);
    CudaErrchkNoCode();
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    printf("intersectSphere: %f\n", time);
    float *it = int_times.data();
    float *ic = int_coords.data();
    CudaErrchk( cudaMemcpy(it, device_time, 2*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(ic, intersect, 6*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaFree(device_time) );
    CudaErrchk( cudaFree(intersect) );
}
