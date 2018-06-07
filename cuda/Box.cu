#include <cfloat>

#include "Box.hpp"
#include "Error.hpp"
#include "Kernels.hpp"

/*void Box::accept(UnaryVisitor &v)
{
    throw "This function is not yet implemented.\n";
}*/

void Box::intersect(float *d_rx, float *d_ry, float *d_rz,
                    float *d_vx, float *d_vy, float *d_vz,
                    const int N, const int blockSize, const int numBlocks,
                    std::vector<float> &int_times, std::vector<float> &int_coords)
{
    float *device_time;
    CudaErrchk( cudaMalloc(&device_time, 6*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(device_time, 6*N, -5);
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
    intersectBox<<<numBlocks, blockSize>>>(d_rx, d_ry, d_rz,
                                           d_vx, d_vy, d_vz,
                                           X, Y, Z,
                                           N, device_time, intersect);
    simplifyTimes<<<numBlocks, blockSize>>>(device_time, N, 6, simp_times);
    CudaErrchkNoCode();
    float *it = int_times.data();
    float *ic = int_coords.data();
    CudaErrchk( cudaMemcpy(it, simp_times, 2*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(ic, intersect, 6*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaFree(device_time) );
    CudaErrchk( cudaFree(intersect) );
    CudaErrchk( cudaFree(simp_times) );
}
