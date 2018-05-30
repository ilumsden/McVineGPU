#include "Cylinder.hpp"
#include "Error.hpp"

void Cylinder::intersect(float *d_rx, float *d_ry, float *d_rz,
                         float *d_vx, float *d_vy, float *d_vz,
                         const int N, const int blockSize, const int numBlocks,
                         std::vector<float> &int_times, std::vector<float> &int_coords)
{
    float *device_time;
    CudaError( cudaMalloc(&device_time, 4*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(device_time, 4*N, -5);
    CudaErrorNoCode();
    float *intersect;
    CudaError( cudaMalloc(&intersect, 6*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(intersect, 6*N, FLT_MAX);
    CudaErrorNoCode();
    float *simp_times;
    CudaError( cudaMalloc(&simp_times, 2*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(simp_times, 2*N, -5);
    CudaErrorNoCode();
    int_times.resize(2*N);
    int_coords.resize(6*N);
    intersectCylinder<<<numBlocks, blockSize>>>(d_rx, d_ry, d_rz,
                                                d_vx, d_vy, d_vz,
                                                radius, height,
                                                N, device_time, intersect);
    simplifyTimes<<<numBlocks, blockSize>>>(device_time, N, 4, simp_times);
    CudaErrorNoCode();
    float *it = int_times.data();
    float *ic = int_coords.data();
    CudaError( cudaMemcpy(it, simp_times, 2*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaError( cudaMemcpy(ic, intersect, 6*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaError( cudaFree(device_time) );
    CudaError( cudaFree(intersect) );
    CudaError( cudaFree(simp_times) );
}
