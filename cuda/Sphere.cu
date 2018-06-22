#include <cfloat>

#include <chrono>

#include "Error.hpp"
#include "Kernels.hpp"
#include "Sphere.hpp"

void Sphere::intersect(//float *d_rx, float *d_ry, float *d_rz,
                       //float *d_vx, float *d_vy, float *d_vz,
                       Vec3<float> *d_origins, Vec3<float> *d_vel,
                       const int N, const int blockSize, const int numBlocks,
                       std::vector<float> &int_times, 
                       std::vector< Vec3<float> > &int_coords)//std::vector<float> &int_coords)
{
    /* The device float array "device_time" is allocated on device, and
     * its elements' values are set to -5.
     * This array will store the times calculated by the
     * intersectSphere kernel.
     * NOTE: Because there are no "sides" to a sphere, this array has
     *       a size of 2*N. As a result, there is no need for the
     *       simplifyTimes kernel or a simp_times float array.
     */
    float *device_time;
    CudaErrchk( cudaMalloc(&device_time, 2*N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(device_time, 2*N, -5);
    CudaErrchkNoCode();
    /* The device float array "intersect" is allocated on device, and
     * its elements' values are set to FLT_MAX.
     * This array will store the intersection coordinates calculated
     * by the intersectSphere kernel.
     */
    //float *intersect;
    Vec3<float> *intersect;
    //CudaErrchk( cudaMalloc(&intersect, 6*N*sizeof(float)) );
    CudaErrchk( cudaMalloc(&intersect, 2*N*sizeof(Vec3<float>)) );
    //initArray<float><<<numBlocks, blockSize>>>(intersect, 6*N, FLT_MAX);
    initArray< Vec3<float> ><<<numBlocks, blockSize>>>(intersect, 2*N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
    CudaErrchkNoCode();
    // These vectors are resized to match the size of the arrays above.
    int_times.resize(2*N);
    //int_coords.resize(6*N);
    int_coords.resize(2*N);
    // This kernel is called to perform the intersection calculation.
    intersectSphere<<<numBlocks, blockSize>>>(//d_rx, d_ry, d_rz,
                                              //d_vx, d_vy, d_vz,
                                              d_origins, d_vel,
                                              radius,
                                              N, device_time, intersect);
    CudaErrchkNoCode();
    /* The data from device_time and intersect is copied into
     * int_times and int_coords respectively.
     */
    float *it = int_times.data();
    //float *ic = int_coords.data();
    Vec3<float> *ic = int_coords.data();
    CudaErrchk( cudaMemcpy(it, device_time, 2*N*sizeof(float), cudaMemcpyDeviceToHost) );
    //CudaErrchk( cudaMemcpy(ic, intersect, 6*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(ic, intersect, 2*N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    /* The device memory allocated at the beginning of the function
     * is freed.
     */
    CudaErrchk( cudaFree(device_time) );
    CudaErrchk( cudaFree(intersect) );
}
