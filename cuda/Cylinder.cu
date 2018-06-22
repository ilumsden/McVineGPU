#include "Cylinder.hpp"
#include "Error.hpp"

void Cylinder::intersect(//float *d_rx, float *d_ry, float *d_rz,
                         //float *d_vx, float *d_vy, float *d_vz,
                         Vec3<float> *d_origins, Vec3<float> *d_vel,
                         const int N, const int blockSize, const int numBlocks,
                         std::vector<float> &int_times, 
                         std::vector< Vec3<float> > &int_coords)//std::vector<float> &int_coords)
{
    /* The device float array "device_time" is allocated on device, and
     * its elements' values are set to -5.
     * This array will store the times calculated by the intersectCylinder
     * kernel.
     */
    float *device_time;
    CudaErrchk( cudaMalloc(&device_time, 4*N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(device_time, 4*N, -5);
    CudaErrchkNoCode();
    /* The device float array "intersect" is allocated on the device, and
     * its elements' values are set to FLT_MAX.
     * This array will store the intersection coordinates calculated
     * by the intersectCylinder kernel.
     */
    //float *intersect;
    Vec3<float> *intersect;
    //CudaErrchk( cudaMalloc(&intersect, 6*N*sizeof(float)) );
    CudaErrchk( cudaMalloc(&intersect, 2*N*sizeof(Vec3<float>)) );
    //initArray<float><<<numBlocks, blockSize>>>(intersect, 6*N, FLT_MAX);
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
    //int_coords.resize(6*N);
    int_coords.resize(2*N);
    // The kernels are called to perform the intersection calculation.
    intersectCylinder<<<numBlocks, blockSize>>>(//d_rx, d_ry, d_rz,
                                                //d_vx, d_vy, d_vz,
                                                d_origins, d_vel,
                                                radius, height,
                                                N, device_time, intersect);
    simplifyTimes<<<numBlocks, blockSize>>>(device_time, N, 4, simp_times);
    CudaErrchkNoCode();
    /* The data from simp_times and intersect is copied into
     * int_times and int_coords respectively.
     */
    float *it = int_times.data();
    //float *ic = int_coords.data();
    Vec3<float> *ic = int_coords.data();
    CudaErrchk( cudaMemcpy(it, simp_times, 2*N*sizeof(float), cudaMemcpyDeviceToHost) );
    //CudaErrchk( cudaMemcpy(ic, intersect, 6*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(ic, intersect, 2*N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    /* The device memory allocated at the beginning of the function
     * is freed.
     */
    CudaErrchk( cudaFree(device_time) );
    CudaErrchk( cudaFree(intersect) );
    CudaErrchk( cudaFree(simp_times) );
}
