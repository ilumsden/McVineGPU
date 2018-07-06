#include <cfloat>

#include "Error.hpp"
#include "Pyramid.hpp"

void Pyramid::exteriorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                const int N, const int blockSize, const int numBlocks,
                                std::vector<float> &int_times, 
                                std::vector< Vec3<float> > &int_coords)
{
    /* The device float array "device_time" is allocated on device, and
     * its elements' values are set to -5.
     * This array will store the times calculated by the intersectPyramid
     * kernel.
     */
    float *device_time;
    CudaErrchk( cudaMalloc(&device_time, 5*N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(device_time, 5*N, -5);
    CudaErrchkNoCode();
    /* The device Vec3<float> array "intersect" is allocated on device, and
     * its elements' values are set to FLT_MAX.
     * This array will store the intersection coordinates calculated
     * by the intersectPyramid kernel.
     */
    Vec3<float> *intersect;
    CudaErrchk( cudaMalloc(&intersect, 2*N*sizeof(Vec3<float>)) );
    initArray< Vec3<float> ><<<numBlocks, blockSize>>>(intersect, 2*N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
    CudaErrchkNoCode();
    /* The device float array "simp_times" is allocated on device, and
     * its elements' values are set to -5.
     * This array will store the output of the simplifyTimes kernel.
     */
    float *simp_times;
    CudaErrchk( cudaMalloc(&simp_times, 2*N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(simp_times, 2*N, -5);
    CudaErrchkNoCode();
    // These vectors are resized to match the size of the arrays above.
    int_times.resize(2*N);
    int_coords.resize(2*N);
    // The kernels are called to perform the intersection calculation.
    intersectPyramid<<<numBlocks, blockSize>>>(d_origins, d_vel,
                                               edgeX, edgeY, height,
                                               N, device_time, intersect);
    simplifyTimes<<<numBlocks, blockSize>>>(device_time, N, 5, 2, simp_times);
    forceIntersectionOrder<<<numBlocks, blockSize>>>(device_time, intersect, N);
    CudaErrchkNoCode();
    /* The data from simp_times and intersect is copied into
     * int_times and int_coords respectively.
     */
    float *it = int_times.data();
    Vec3<float> *ic = int_coords.data();
    CudaErrchk( cudaMemcpy(it, simp_times, 2*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(ic, intersect, 2*N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    /* The device memory allocated at the beginning of the function
     * is freed.
     */
    CudaErrchk( cudaFree(device_time) );
    CudaErrchk( cudaFree(intersect) );
    CudaErrchk( cudaFree(simp_times) );
}

void Pyramid::interiorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                const int N, const int blockSize, const int numBlocks,
                                std::vector<float> &int_times, 
                                std::vector< Vec3<float> > &int_coords)
{
    /* The device float array "device_time" is allocated on device, and
     * its elements' values are set to -5.
     * This array will store the times calculated by the intersectPyramid
     * kernel.
     */
    float *device_time;
    CudaErrchk( cudaMalloc(&device_time, 5*N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(device_time, 5*N, -5);
    CudaErrchkNoCode();
    /* The device Vec3<float> array "intersect" is allocated on device, and
     * its elements' values are set to FLT_MAX.
     * This array will store the intersection coordinates calculated
     * by the intersectPyramid kernel.
     */
    Vec3<float> *intersect;
    CudaErrchk( cudaMalloc(&intersect, 2*N*sizeof(Vec3<float>)) );
    initArray< Vec3<float> ><<<numBlocks, blockSize>>>(intersect, 2*N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
    CudaErrchkNoCode();
    /* The device float array "simp_times" is allocated on device, and
     * its elements' values are set to -5.
     * This array will store the output of the simplifyTimes kernel.
     */
    float *simp_times;
    CudaErrchk( cudaMalloc(&simp_times, N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(simp_times, N, -5);
    CudaErrchkNoCode();
    /* The Vec3<float> array "simp_times" is allocated on device, and
     * its elements' values are set to FLT_MAX.
     * This array will store the output of the simplifyPoints kernel.
     */
    Vec3<float> *simp_int;
    CudaErrchk( cudaMalloc(&simp_int, N*sizeof(Vec3<float>)) );
    initArray< Vec3<float> ><<<numBlocks, blockSize>>>(simp_int, N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
    CudaErrchkNoCode();
    // These vectors are resized to match the size of the arrays above.
    int_times.resize(N);
    int_coords.resize(N);
    // The kernels are called to perform the intersection calculation.
    intersectPyramid<<<numBlocks, blockSize>>>(d_origins, d_vel,
                                               edgeX, edgeY, height,
                                               N, device_time, intersect);
    simplifyTimes<<<numBlocks, blockSize>>>(device_time, N, 5, 1, simp_times);
    simplifyPoints<<<numBlocks, blockSize>>>(intersect, N, 2, 1, simp_int);
    CudaErrchkNoCode();
    /* The data from simp_times and intersect is copied into
     * int_times and int_coords respectively.
     */
    float *it = int_times.data();
    Vec3<float> *ic = int_coords.data();
    CudaErrchk( cudaMemcpy(it, simp_times, N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(ic, simp_int, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    /* The device memory allocated at the beginning of the function
     * is freed.
     */
    CudaErrchk( cudaFree(device_time) );
    CudaErrchk( cudaFree(intersect) );
    CudaErrchk( cudaFree(simp_times) );
    CudaErrchk( cudaFree(simp_int) );
}
