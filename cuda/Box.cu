#include <cfloat>

#include "Box.hpp"
#include "Error.hpp"

void Box::exteriorIntersect(std::vector<Vec3<float>*> &d_origins,
                            std::vector<Vec3<float>*> &d_vel,
                            const int blockSize, 
                            const std::vector<int> &numBlocks,
                            const std::vector<int> &steps,
                            std::vector<float> &int_times,
                            std::vector< Vec3<float> > &int_coords)
{
    /* The device float array "device_time" is allocated on device, and
     * its elements' values are set to -5.
     * This array will store the times calculated by the intersectBox
     * kernel.
     */
    /*float *device_time;
    CudaErrchk( cudaMalloc(&device_time, 6*N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(device_time, 6*N, -5);
    CudaErrchkNoCode();*/
    std::vector<float*> device_time;
    device_time.resize(numBlocks.size());
    for (int i = 0; i < (int)(numBlocks.size()); i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(device_time[i]), 6*(steps[i+1]-steps[i])*sizeof(float)) );
        initArray<float><<<numBlocks[i], blockSize>>>(device_time[i], 6*(steps[i+1]-steps[i]), -5);
        CudaErrchkNoCode();
    }
    /* The device Vec3<float> array "intersect" is allocated on device, and
     * its elements' values are set to FLT_MAX.
     * This array will store the intersection coordinates calculated 
     * by the intersectBox kernel.
     */
    /*Vec3<float> *d_intersect;
    CudaErrchk( cudaMalloc(&d_intersect, 2*N*sizeof(Vec3<float>)) );
    initArray< Vec3<float> ><<<numBlocks, blockSize>>>(d_intersect, 2*N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
    CudaErrchkNoCode();*/
    std::vector<Vec3<float>*> d_intersect;
    d_intersect.resize(numBlocks.size());
    for (int i = 0; i < (int)(numBlocks.size()); i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(d_intersect[i]), 2*(steps[i+1]-steps[i])*sizeof(Vec3<float>)) );
        initArray< Vec3<float> ><<<numBlocks[i], blockSize>>>(d_intersect[i], 2*(steps[i+1]-steps[i]), Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX));
        CudaErrchkNoCode();
    }
    /* The device float array "simp_times" is allocated on device, and
     * its elements' values are set to -5.
     * This array will store the output of the simplifyTimes kernel.
     */
    /*float *simp_times;
    CudaErrchk( cudaMalloc(&simp_times, 2*N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(simp_times, 2*N, -5);
    CudaErrchkNoCode();*/
    std::vector<float*> simp_times;
    simp_times.resize(numBlocks.size());
    for (int i = 0; i < (int)(numBlocks.size()); i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(simp_times[i]), 2*(steps[i+1]-steps[i])*sizeof(float)) );
        initArray<float><<<numBlocks[i], blockSize>>>(simp_times[i], 2*(steps[i+1]-steps[i]), -5);
        CudaErrchkNoCode();
    }
    /*float *d_data;
    CudaErrchk( cudaMalloc(&d_data, 3*sizeof(float)) );
    CudaErrchk( cudaMemcpy(d_data, data, 3*sizeof(float), cudaMemcpyHostToDevice) );*/
    std::vector<float*> d_data;
    d_data.resize(numBlocks.size());
    for (int i = 0; i < (int)(numBlocks.size()); i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(d_data[i]), 3*sizeof(float)) );
        CudaErrchk( cudaMemcpy(d_data[i], data, 3*sizeof(float), cudaMemcpyHostToDevice) );
    }
    // These vectors are resized to match the size of the arrays above.
    int_times.resize(2*N);
    int_coords.resize(2*N);
    // The kernels are called to perform the intersection calculation.
    /*intersectBox<<<numBlocks, blockSize>>>(d_origins,
                                           d_vel,
                                           X, Y, Z,
                                           N, device_time, intersect);*/
    /*intersect<<<numBlocks, blockSize>>>(interKeyDict[type],
                                        d_origins, d_vel, d_data, N,
                                        device_time, d_intersect);*/
    //simplifyTimes<<<numBlocks, blockSize>>>(device_time, N, 6, 2, simp_times);
    /*simplifyTimePointPairs<<<numBlocks, blockSize>>>(device_time, 
                                                     d_intersect,
                                                     N, 6, 2, 2,
                                                     simp_times,
                                                     d_intersect);
    forceIntersectionOrder<<<numBlocks, blockSize>>>(simp_times, d_intersect, N);
    CudaErrchkNoCode();*/
    /* The data from simp_times and intersect is copied into
     * int_times and int_coords respectively.
     */
    /*float *it = int_times.data();
    Vec3<float> *ic = int_coords.data();
    CudaErrchk( cudaMemcpy(it, simp_times, 2*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(ic, d_intersect, 2*N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );*/
    float *it = int_times.data();
    Vec3<float> *ic = int_coords.data();
    for (int i = 0; i < (int)(numBlocks.size()); i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        intersect<<<numBlocks[i], blockSize>>>(interKeyDict[type],
                                               d_origins[i], d_vel[i],
                                               d_data[i], 
                                               (steps[i+1]-steps[i]),
                                               device_time[i], d_intersect[i]);
        simplifyTimePointPairs<<<numBlocks[i], blockSize>>>(device_time[i],
                                                            d_intersect[i],
                                                            (steps[i+1]-steps[i]),
                                                            6, 2, 2,
                                                            simp_times[i],
                                                            d_intersect[i]);
        forceIntersectionOrder<<<numBlocks[i], blockSize>>>(simp_times[i],
                                                            d_intersect[i],
                                                            (steps[i+1]-steps[i]));
        CudaErrchkNoCode();
        CudaErrchk( cudaMemcpyAsync(&(it[steps[i]]), simp_times[i], 2*(steps[i+1]-steps[i])*sizeof(float), cudaMemcpyDeviceToHost) );
        CudaErrchk( cudaMemcpyAsync(&(ic[steps[i]]), d_intersect[i], 2*(steps[i+1]-steps[i])*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    }
    /* The device memory allocated at the beginning of the function
     * is freed.
     */
    for (int i = 0; i < (int)(numBlocks.size()); i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaFree(device_time[i]) );
        CudaErrchk( cudaFree(d_intersect[i]) );
        CudaErrchk( cudaFree(simp_times[i]) );
        CudaErrchk( cudaFree(d_data[i]) );
    }
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
    /*float *device_time;
    CudaErrchk( cudaMalloc(&device_time, 6*N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(device_time, 6*N, -5);
    CudaErrchkNoCode();*/
    std::vector<float*> device_time;
    device_time.resize(numBlocks.size());
    for (int i = 0; i < (int)(numBlocks.size()); i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(device_time[i]), 6*(steps[i+1]-steps[i])*sizeof(float)) );
        initArray<float><<<numBlocks[i], blockSize>>>(device_time[i], 6*(steps[i+1]-steps[i]), -5);
        CudaErrchkNoCode();
    }
    /* The device Vec3<float> array "intersect" is allocated on device, and
     * its elements' values are set to FLT_MAX.
     * This array will store the intersection coordinates calculated 
     * by the intersectBox kernel.
     */
    /*Vec3<float> *d_intersect;
    CudaErrchk( cudaMalloc(&d_intersect, 2*N*sizeof(Vec3<float>)) );
    initArray< Vec3<float> ><<<numBlocks, blockSize>>>(d_intersect, 2*N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
    CudaErrchkNoCode();*/
    std::vector<Vec3<float>*> d_intersect;
    d_intersect.resize(numBlocks.size());
    for (int i = 0; i < (int)(numBlocks.size()); i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(d_intersect[i]), 2*(steps[i+1]-steps[i])*sizeof(Vec3<float>)) );
        initArray< Vec3<float> ><<<numBlocks[i], blockSize>>>(d_intersect[i], 2*(steps[i+1]-steps[i]), Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX));
        CudaErrchkNoCode();
    }
    /* The device float array "simp_times" is allocated on device, and
     * its elements' values are set to -5.
     * This array will store the output of the simplifyTimes kernel.
     */
    /*float *simp_times;
    CudaErrchk( cudaMalloc(&simp_times, N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(simp_times, N, -5);
    CudaErrchkNoCode();*/
    std::vector<float*> simp_times;
    simp_times.resize(numBlocks.size());
    for (int i = 0; i < (int)(numBlocks.size()); i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(simp_times[i]), (steps[i+1]-steps[i])*sizeof(float)) );
        initArray<float><<<numBlocks[i], blockSize>>>(simp_times[i], (steps[i+1]-steps[i]), -5);
        CudaErrchkNoCode();
    }
    /* The Vec3<float> array "simp_int" is allocated on the device, and
     * its elements' values are set to FLT_MAX.
     * This array will store the output of the simplifyPoints kernel.
     */
    /*Vec3<float> *simp_int;
    CudaErrchk( cudaMalloc(&simp_int, N*sizeof(Vec3<float>)) );
    initArray< Vec3<float> ><<<numBlocks, blockSize>>>(simp_int, N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
    CudaErrchkNoCode();*/
    std::vector<Vec3<float>*> simp_int;
    simp_int.resize(numBlocks.size());
    for (int i = 0; i < (int)(numBlocks.size()); i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(simp_int[i]), (steps[i+1]-steps[i])*sizeof(Vec3<float>)) );
        initArray< Vec3<float> ><<<numBlocks[i], blockSize>>>(simp_int[i], (steps[i+1]-steps[i]), Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX));
        CudaErrchkNoCode();
    }
    /*float *d_data;
    CudaErrchk( cudaMalloc(&d_data, 3*sizeof(float)) );
    CudaErrchk( cudaMemcpy(d_data, data, 3*sizeof(float), cudaMemcpyHostToDevice) );*/
    std::vector<float*> d_data;
    d_data.resize(numBlocks.size());
    for (int i = 0; i < (int)(numBlocks.size()); i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(d_data[i]), 3*sizeof(float)) );
        CudaErrchk( cudaMemcpy(d_data[i], data, 3*sizeof(float), cudaMemcpyHostToDevice) );
    }
    // These vectors are resized to match the size of the arrays above.
    int_times.resize(N);
    int_coords.resize(N);
    float *it = int_times.data();
    Vec3<float> *ic = int_coord.data();
    // The kernels are called to perform the intersection calculation.
    /*intersectBox<<<numBlocks, blockSize>>>(d_origins,
                                           d_vel,
                                           X, Y, Z,
                                           N, device_time, intersect);*/
    /*intersect<<<numBlocks, blockSize>>>(interKeyDict[type],
                                        d_origins, d_vel, d_data, N,
                                        device_time, d_intersect);*/
    //simplifyTimes<<<numBlocks, blockSize>>>(device_time, N, 6, 1, simp_times);
    //simplifyPoints<<<numBlocks, blockSize>>>(intersect, N, 2, 1, simp_int);
    /*simplifyTimePointPairs<<<numBlocks, blockSize>>>(device_time,
                                                     d_intersect,
                                                     N, 6, 2, 1,
                                                     simp_times,
                                                     simp_int);
    CudaErrchkNoCode();*/
    /* The data from simp_times and intersect is copied into
     * int_times and int_coords respectively.
     */
    /*float *it = int_times.data();
    Vec3<float> *ic = int_coords.data();
    CudaErrchk( cudaMemcpy(it, simp_times, N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(ic, simp_int, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );*/
    for (int i = 0; i < (int)(numBlocks.size()); i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        intersect<<<numBlocks[i], blockSize>>>(interKeyDict[type],
                                               d_origins[i], d_vel[i],
                                               d_data[i],
                                               (steps[i+1]-steps[i]),
                                               device_time[i], d_intersect[i]);
        simplifyTimePointPairs<<<numBlocks[i], blockSize>>>(device_time[i],
                                                            d_intersect[i],
                                                            (steps[i+1]-steps[i]),
                                                            6, 2, 1,
                                                            simp_times[i],
                                                            simp_int[i];
        CudaErrchkNoCode();
        CudaErrchk( cudaMemcpyAsync(&(it[steps[i]]), simp_times[i], (steps[i+1]-steps[i])*sizeof(float), cudaMemcpyDeviceToHost) );
        CudaErrchk( cudaMemcpyAsync(&(ic[steps[i]]), simp_int[i], (steps[i+1]-steps[i])*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    }
    /* The device memory allocated at the beginning of the function
     * is freed.
     */
    for (int i = 0; i < (int)(numBlocks.size()); i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaFree(device_time[i]) );
        CudaErrchk( cudaFree(d_intersect[i]) );
        CudaErrchk( cudaFree(simp_times[i]) );
        CudaErrchk( cudaFree(simp_int[i]) );
        CudaErrchk( cudaFree(d_data[i]) );
    }
}
