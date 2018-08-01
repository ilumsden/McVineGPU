#include <cfloat>

#include <chrono>

#include "Error.hpp"
#include "Sphere.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace composite
        {

            void Sphere::exteriorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                           const int N, const int blockSize, const int numBlocks,
                                           std::vector<float> &int_times, 
                                           std::vector< Vec3<float> > &int_coords)
            {
                namespace kernels = mcvine::gpu::kernels;
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
                kernels::initArray<float><<<numBlocks, blockSize>>>(device_time, 2*N, -5);
                CudaErrchkNoCode();
                /* The device Vec3<float> array "intersect" is allocated on device, and
                 * its elements' values are set to FLT_MAX.
                 * This array will store the intersection coordinates calculated
                 * by the intersectSphere kernel.
                 */
                Vec3<float> *d_intersect;
                CudaErrchk( cudaMalloc(&d_intersect, 2*N*sizeof(Vec3<float>)) );
                kernels::initArray< Vec3<float> ><<<numBlocks, blockSize>>>(d_intersect, 2*N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
                CudaErrchkNoCode();
                float *d_data;
                CudaErrchk( cudaMalloc(&d_data, sizeof(float)) );
                CudaErrchk( cudaMemcpy(d_data, data, sizeof(float), cudaMemcpyHostToDevice) );
                // These vectors are resized to match the size of the arrays above.
                int_times.resize(2*N);
                int_coords.resize(2*N);
                /*intersectSphere<<<numBlocks, blockSize>>>(d_origins, d_vel,
                                                          radius,
                                                          N, device_time, intersect);*/
                kernels::intersect<<<numBlocks, blockSize>>>(interKeyDict[type],
                                                             d_origins, d_vel, d_data, N,
                                                             device_time, d_intersect);
                kernels::forceIntersectionOrder<<<numBlocks, blockSize>>>(device_time, d_intersect, N);
                CudaErrchkNoCode();
                /* The data from device_time and intersect is copied into
                 * int_times and int_coords respectively.
                 */
                float *it = int_times.data();
                Vec3<float> *ic = int_coords.data();
                CudaErrchk( cudaMemcpy(it, device_time, 2*N*sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(ic, d_intersect, 2*N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                /* The device memory allocated at the beginning of the function
                 * is freed.
                 */
                CudaErrchk( cudaFree(device_time) );
                CudaErrchk( cudaFree(d_intersect) );
                CudaErrchk( cudaFree(d_data) );
            }

            void Sphere::interiorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                           const int N, const int blockSize, const int numBlocks,
                                           std::vector<float> &int_times, 
                                           std::vector< Vec3<float> > &int_coords)
            {
                namespace kernels = mcvine::gpu::kernels;
                /* The device float array "device_time" is allocated on device, and
                 * its elements' values are set to -5.
                 * This array will store the times calculated by the
                 * intersectSphere kernel.
                 */
                float *device_time;
                CudaErrchk( cudaMalloc(&device_time, 2*N*sizeof(float)) );
                kernels::initArray<float><<<numBlocks, blockSize>>>(device_time, 2*N, -5);
                CudaErrchkNoCode();
                /* The device Vec3<float> array "intersect" is allocated on device, and
                 * its elements' values are set to FLT_MAX.
                 * This array will store the intersection coordinates calculated
                 * by the intersectSphere kernel.
                 */
                Vec3<float> *d_intersect;
                CudaErrchk( cudaMalloc(&d_intersect, 2*N*sizeof(Vec3<float>)) );
                kernels::initArray< Vec3<float> ><<<numBlocks, blockSize>>>(d_intersect, 2*N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
                CudaErrchkNoCode();
                /* The float array "simp_times" is allocated on the deivce, and
                 * its elements' values are set to -5.
                 * This array will store the output of the simplifyTimes kernel.
                 */
                float *simp_times;
                CudaErrchk( cudaMalloc(&simp_times, N*sizeof(float)) );
                kernels::initArray<float><<<numBlocks, blockSize>>>(simp_times, N, -5);
                CudaErrchkNoCode();
                /* The Vec3<float> array "simp_int" is allocated on the deivce, and
                 * its elements' values are set to FLT_MAX.
                 * This array will store the output of the simplifyPoints kernel.
                 */
                Vec3<float> *simp_int;
                CudaErrchk( cudaMalloc(&simp_int, N*sizeof(Vec3<float>)) );
                kernels::initArray< Vec3<float> ><<<numBlocks, blockSize>>>(simp_int, N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
                CudaErrchkNoCode();
                float *d_data;
                CudaErrchk( cudaMalloc(&d_data, sizeof(float)) );
                CudaErrchk( cudaMemcpy(d_data, data, sizeof(float), cudaMemcpyHostToDevice) );
                // These vectors are resized to match the size of the arrays above.
                int_times.resize(N);
                int_coords.resize(N);
                /*intersectSphere<<<numBlocks, blockSize>>>(d_origins, d_vel,
                                                          radius,
                                                          N, device_time, intersect);*/
                kernels::intersect<<<numBlocks, blockSize>>>(interKeyDict[type],
                                                             d_origins, d_vel, d_data, N,
                                                             device_time, d_intersect);
                kernels::simplifyTimePointPairs<<<numBlocks, blockSize>>>(
                    device_time,
                    d_intersect,
                    N, 2, 2, 1,
                    simp_times,
                    simp_int);
                CudaErrchkNoCode();
                /* The data from device_time and intersect is copied into
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
                CudaErrchk( cudaFree(d_intersect) );
                CudaErrchk( cudaFree(simp_times) );
                CudaErrchk( cudaFree(simp_int) );
                CudaErrchk( cudaFree(d_data) );
            }

        }

    }

}
