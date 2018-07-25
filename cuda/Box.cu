#include <cfloat>

#include "Box.hpp"
#include "Error.hpp"

void Box::exteriorIntersect(Vec3<float> *d_origins,
                            Vec3<float> *d_vel,
                            const int N, const int blockSize, const int numBlocks,
                            std::vector<float> &int_times,
                            std::vector< Vec3<float> > &int_coords)
{
    /* The device float array "device_time" is allocated on device, and
     * its elements' values are set to -5.
     * This array will store the times calculated by the intersectBox
     * kernel.
     */
    float *device_time;
    CudaErrchk( cudaMalloc(&device_time, 6*N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(device_time, 6*N, -5);
    CudaErrchkNoCode();
    /* The device Vec3<float> array "intersect" is allocated on device, and
     * its elements' values are set to FLT_MAX.
     * This array will store the intersection coordinates calculated 
     * by the intersectBox kernel.
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
    float *d_data;
    CudaErrchk( cudaMalloc(&d_data, 3*sizeof(float)) );
    CudaErrchk( cudaMemcpy(d_data, data, 3*sizeof(float), cudaMemcpyHostToDevice) );
    // These vectors are resized to match the size of the arrays above.
    int_times.resize(2*N);
    int_coords.resize(2*N);
    // The kernels are called to perform the intersection calculation.
    /*intersectBox<<<numBlocks, blockSize>>>(d_origins,
                                           d_vel,
                                           X, Y, Z,
                                           N, device_time, intersect);*/
    intersect<<<numBlocks, blockSize>>>(std::get<0>(funcPtrDict[type]),
                                        d_origins, d_vel, d_data, N,
                                        std::get<1>(funcPtrDict[type]),
                                        device_time, intersect);
    //simplifyTimes<<<numBlocks, blockSize>>>(device_time, N, 6, 2, simp_times);
    simplifyTimePointPairs<<<numBlocks, blockSize>>>(device_time, 
                                                     intersect,
                                                     N, 6, 2, 2,
                                                     simp_times,
                                                     intersect);
    forceIntersectionOrder<<<numBlocks, blockSize>>>(simp_times, intersect, N);
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
    CudaErrchk( cudaFree(d_data) );
}

void Box::interiorIntersect(Vec3<float> *d_origins,
                            Vec3<float> *d_vel,
                            const int N, const int blockSize, const int numBlocks,
                            std::vector<float> &int_times,
                            std::vector< Vec3<float> > &int_coords)
{
#if defined(INTERIORTEST)
    std::vector< Vec3<float> > exit_coords;
    std::vector<float> exit_times;
    exit_coords.resize(2*N);
    exit_times.resize(6*N);
#endif
    /* The device float array "device_time" is allocated on device, and
     * its elements' values are set to -5.
     * This array will store the times calculated by the intersectBox
     * kernel.
     */
    float *device_time;
    CudaErrchk( cudaMalloc(&device_time, 6*N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(device_time, 6*N, -5);
    CudaErrchkNoCode();
    /* The device Vec3<float> array "intersect" is allocated on device, and
     * its elements' values are set to FLT_MAX.
     * This array will store the intersection coordinates calculated 
     * by the intersectBox kernel.
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
    /* The Vec3<float> array "simp_int" is allocated on the device, and
     * its elements' values are set to FLT_MAX.
     * This array will store the output of the simplifyPoints kernel.
     */
    Vec3<float> *simp_int;
    CudaErrchk( cudaMalloc(&simp_int, N*sizeof(Vec3<float>)) );
    initArray< Vec3<float> ><<<numBlocks, blockSize>>>(simp_int, N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
    CudaErrchkNoCode();
    float *d_data;
    CudaErrchk( cudaMalloc(&d_data, 3*sizeof(float)) );
    CudaErrchk( cudaMemcpy(d_data, data, 3*sizeof(float), cudaMemcpyHostToDevice) );
    // These vectors are resized to match the size of the arrays above.
    int_times.resize(N);
    int_coords.resize(N);
    // The kernels are called to perform the intersection calculation.
    /*intersectBox<<<numBlocks, blockSize>>>(d_origins,
                                           d_vel,
                                           X, Y, Z,
                                           N, device_time, intersect);*/
    intersect<<<numBlocks, blockSize>>>(std::get<0>(funcPtrDict[type]),
                                        d_origins, d_vel, d_data, N,
                                        std::get<1>(funcPtrDict[type]),
                                        device_time, intersect);
#if defined(INTERIORTEST)
    Vec3<float> *ec = exit_coords.data();
    float *et = exit_times.data();
    CudaErrchk( cudaMemcpy(ec, intersect, 2*N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(et, device_time, 6*N*sizeof(float), cudaMemcpyDeviceToHost) );
    std::fstream fout;
    fout.open("interiorTest.txt", std::ios::out);
    if (!fout.is_open())
    {
        std::cerr << "interiorTest.txt could not be openned.\n";
        exit(-2);
    }
    fout << std::setw(8) << std::right << "Times:" << "Coords:" << "\n";
    for (int i = 0; i < 6*N; i++)
    {
        if (i%6 == 0)
        {
            fout << "\n";
        }
        int ind = i/3;
        fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
             << exit_times[i] << " " << exit_coords[ind][i - (ind*3)] << "\n";
    }
    fout.close();
#endif
    //simplifyTimes<<<numBlocks, blockSize>>>(device_time, N, 6, 1, simp_times);
    //simplifyPoints<<<numBlocks, blockSize>>>(intersect, N, 2, 1, simp_int);
    simplifyTimePointPairs<<<numBlocks, blockSize>>>(device_time,
                                                     intersect,
                                                     N, 6, 2, 1,
                                                     simp_times,
                                                     simp_int);
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
    CudaErrchk( cudaFree(d_data) );
}
