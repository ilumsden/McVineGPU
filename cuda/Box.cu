#include <cfloat>

#include "Box.hpp"
#include "Error.hpp"
#include "Kernels.hpp"

Box::Box(const double a, const double b, const double c)
{
    x = a;
    y = b;
    z = c;
    xmin = -(a/2);
    xmax = a/2;
    ymin = -(b/2);
    ymax = b/2;
    zmin = -(c/2);
    zmax = c/2;
}

Box::Box(const double amin, const double bmin, const double cmin,
         const double amax, const double bmax, const double cmax)
{
    xmin = amin;
    ymin = bmin;
    zmin = cmin;
    xmax = amax;
    ymax = bmax;
    zmax = cmax;
    x = amax - amin;
    y = bmax - bmin;
    z = cmax - cmin;
}

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
    CudaError( cudaMalloc(&device_time, 6*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(device_time, 6*N, -5);
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
    intersectBox<<<numBlocks, blockSize>>>(d_rx, d_ry, d_rz,
                                           d_vx, d_vy, d_vz,
                                           xmin, xmax,
                                           ymin, ymax,
                                           zmin, zmax,
                                           N, device_time, intersect);
    simplifyTimes<<<numBlocks, blockSize>>>(device_time, N, 6, simp_times);
    CudaErrorNoCode();
    float *it = int_times.data();
    float *ic = int_coords.data();
    CudaError( cudaMemcpy(it, simp_times, 2*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaError( cudaMemcpy(ic, intersect, 6*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaError( cudaFree(device_time) );
    CudaError( cudaFree(intersect) );
    CudaError( cudaFree(simp_times) );
}
