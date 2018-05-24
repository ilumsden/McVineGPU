#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <chrono>

#include "CudaDriver.hpp"
#include "Kernels.hpp"
#include "Error.hpp"

CudaDriver::CudaDriver(const std::vector< std::shared_ptr<Ray> > &rays, int bS)
{ 
    // N is the number of rays considered
    N = (int)(rays.size());
    // Calculates the thread and block parameters for CUDA
    blockSize = bS;
    numBlocks = (N + blockSize - 1) / blockSize;
    printf("blockSize = %i\nnumBlocks = %i\n", blockSize, numBlocks);
    /* Allocates both host and device memory for the float arrays that
     * will be used to store the data passed to the CUDA functions.
     */
    rx = (float*)malloc(N*sizeof(float));
    CudaError( cudaMalloc(&d_rx, N*sizeof(float)) );
    ry = (float*)malloc(N*sizeof(float));
    CudaError( cudaMalloc(&d_ry, N*sizeof(float)) );
    rz = (float*)malloc(N*sizeof(float));
    CudaError( cudaMalloc(&d_rz, N*sizeof(float)) );
    vx = (float*)malloc(N*sizeof(float));
    CudaError( cudaMalloc(&d_vx, N*sizeof(float)) );
    vy = (float*)malloc(N*sizeof(float));
    CudaError( cudaMalloc(&d_vy, N*sizeof(float)) );
    vz = (float*)malloc(N*sizeof(float));
    CudaError( cudaMalloc(&d_vz, N*sizeof(float)) );
    // Copies the data from the rays to the CUDA-compatible arrays.
    int c = 0;
    for (auto ray : rays)
    {
        rx[c] = (float)(ray->x);
        ry[c] = (float)(ray->y);
        rz[c] = (float)(ray->z);
        vx[c] = (float)(ray->vx);
        vy[c] = (float)(ray->vy);
        vz[c] = (float)(ray->vz);
        c++;
    }
    CudaError( cudaMemcpy(d_rx, rx, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaError( cudaMemcpy(d_ry, ry, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaError( cudaMemcpy(d_rz, rz, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaError( cudaMemcpy(d_vx, vx, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaError( cudaMemcpy(d_vy, vy, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaError( cudaMemcpy(d_vz, vz, N*sizeof(float), cudaMemcpyHostToDevice) );
}

CudaDriver::~CudaDriver()
{
    free(rx);
    free(ry);
    free(rz);
    free(vx);
    free(vy);
    free(vz);
    CudaError( cudaFree(d_rx) );
    CudaError( cudaFree(d_ry) );
    CudaError( cudaFree(d_rz) );
    CudaError( cudaFree(d_vx) );
    CudaError( cudaFree(d_vy) );
    CudaError( cudaFree(d_vz) );
}

/* This is the host-side driver function for setting up the data for the
 * `intersectRectangle` function above, calling said function, and parsing
 * the returned data.
 */
void CudaDriver::handleRectIntersect(std::shared_ptr<Box> &b, 
                                     std::vector<float> &host_time,
                                     std::vector<float> &int_coords)
{
    /* Creates a float array in host and device memory to store the 
     * results of the CUDA calculations. The initial value of each element
     * is -5.
     */
    //auto start = std::chrono::steady_clock::now();
    float *device_time;
    CudaError( cudaMalloc(&device_time, 6*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(device_time, 6*N, -5);
    CudaErrorNoCode();
    /*auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    printf("Initialize device_time: %f\n", time);*/
    //start = std::chrono::steady_clock::now();
    float *intersect;
    CudaError( cudaMalloc(&intersect, 6*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(intersect, 6*N, FLT_MAX);
    CudaErrorNoCode();
    /*stop = std::chrono::steady_clock::now();
    time = std::chrono::duration<double>(stop - start).count();
    printf("Initialize intersect: %f\n", time);*/
    float *simp_times;
    CudaError( cudaMalloc(&simp_times, 2*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(simp_times, 2*N, -5);
    CudaErrorNoCode();
    host_time.resize(2*N);
    int_coords.resize(6*N);
    // Starts the CUDA code
    //start = std::chrono::steady_clock::now();
    intersectRectangle<<<numBlocks, blockSize>>>(d_rx, d_ry, d_rz, d_vx, d_vy, d_vz, b->x, b->y, b->z, N, device_time, intersect);
    CudaErrorNoCode();
    simplifyTimes<<<numBlocks, blockSize>>>(device_time, N, 6, simp_times);
    CudaErrorNoCode();
    /*stop = std::chrono::steady_clock::now();
    time = std::chrono::duration<double>(stop - start).count();
    printf("intersectRectangle: %f\n", time);*/
    // Halts CPU progress until the CUDA code has finished
    float *ht = host_time.data();
    float *ic = int_coords.data();
    CudaError( cudaMemcpy(ht, simp_times, 2*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaError( cudaMemcpy(ic, intersect, 6*N*sizeof(float), cudaMemcpyDeviceToHost) );
    // Opens a file stream and prints the relevant data to time.txt
    // NOTE: this is for debugging purposes only. This will be removed later.
    std::fstream fout;
    fout.open("time.txt", std::ios::out);
    if (!fout.is_open())
    {
        std::cerr << "time.txt could not be opened.\n";
        exit(-1);
    }
    for (int i = 0; i < (int)(int_coords.size()); i++)
    {
        if (i % 6 == 0)
        {
            int ind = i/6;
            fout << "\n";
            fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                 << rx[ind] << " " << ry[ind] << " " << rz[ind] << " || "
                 << vx[ind] << " " << vy[ind] << " " << vz[ind] << " | "
                 << host_time[i/3] << " / " << int_coords[i] << "\n";
        }
        else if (i % 6 == 1)
        {
            std::string buf = "        ";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                 << std::fixed << std::setprecision(5) << std::setw(8) << std::right << host_time[(i/3)+1] << " / " << int_coords[i] << "\n";
        }
        else
        {
            std::string buf = "        ";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                 << std::fixed << std::setprecision(5) << std::setw(8) << std::right << buf << " / " << int_coords[i] << "\n";
        }
    }
    // Closes the file stream
    fout.close();
    // Frees up the memory allocated by the cudaMallocManaged calls above.
    CudaError( cudaFree(device_time) );
    CudaError( cudaFree(intersect) );
    CudaError( cudaFree(simp_times) );
    return;
}

void CudaDriver::findScatteringSites(std::shared_ptr< Box> &b,
                                     const std::vector<float> &int_times, 
                                     const std::vector<float> &int_coords,
                                     std::vector<float> &sites)
{
    int tsize = (int)(int_times.size());
    int csize = (int)(int_coords.size());
    float *ts, *inters;
    CudaError( cudaMalloc(&ts, 2*N*sizeof(float)) );
    CudaError( cudaMalloc(&inters, 6*N*sizeof(float)) );
    CudaError( cudaMemcpy(ts, int_times.data(), 2*N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaError( cudaMemcpy(inters, int_coords.data(), 6*N*sizeof(float), cudaMemcpyHostToDevice) );
    float *pos;
    CudaError( cudaMalloc(&pos, 3*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(pos, 3*N, FLT_MAX);
    CudaErrorNoCode();
    sites.resize(3*N);
    curandState *state;
    //printf("N*size(curandState) = %i\n", (int)(numBlocks*blockSize*sizeof(curandState)));
    //printf("N*size(float) = %i\n", (int)(N*sizeof(float)));
    CudaError( cudaMalloc(&state, numBlocks*sizeof(curandState)) );
    printf("Pre Rand Prep\n");
    auto start = std::chrono::steady_clock::now();
    prepRand<<<numBlocks, blockSize>>>(state, time(NULL));
    cudaDeviceSynchronize();
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    printf("Rand Prep Complete\n    Summary: Time = %f\n", time);
    printf("Pre Kernel\n");
    calcScatteringSites<<<numBlocks, blockSize>>>(d_rx, d_ry, d_rz, d_vx, d_vy, d_vz, b->x, b->y, b->z, ts, inters, pos, state, N);
    CudaErrorNoCode();
    cudaDeviceSynchronize();
    printf("Post Kernel\n");
    float* s = sites.data();
    CudaError( cudaMemcpy(s, pos, 3*N*sizeof(float), cudaMemcpyDeviceToHost) );
    printf("Post Memcpy\n");
    std::fstream fout;
    fout.open("scatteringSites.txt", std::ios::out);
    if (!fout.is_open())
    {
        std::cerr << "scatteringSites.txt could not be opened.\n";
        exit(-2);
    }
    for (int i = 0; i < (int)(sites.size()); i++)
    {
        if (i % 3 == 0)
        {
            int ind = i/3;
            fout << "\n";
            fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                 << rx[ind] << " " << ry[ind] << " " << rz[ind] << " || "
                 << vx[ind] << " " << vy[ind] << " " << vz[ind] << " | "
                 << sites[i] << "\n";
        }
        else
        {
            std::string buf = "        ";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
<< std::fixed << std::setprecision(5) << std::setw(8) << std::right << sites[i] << "\n";
        }
    }
    fout.close();
    cudaFree(ts);
    cudaFree(inters);
    cudaFree(pos);
    cudaFree(state);
    return;
}

// A simple wrapper of the cudaHandler function
void CudaDriver::runCalculations(std::shared_ptr<Box> &b)
{
    std::vector<float> int_times;
    std::vector<float> int_coords;
    auto start = std::chrono::steady_clock::now();
    handleRectIntersect(b, int_times, int_coords);
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    printf("handleRectIntersect: %f\n", time);
    std::vector<float> scattering_sites;
    start = std::chrono::steady_clock::now();
    findScatteringSites(b, int_times, int_coords, scattering_sites);
    stop = std::chrono::steady_clock::now();
    time = std::chrono::duration<double>(stop - start).count();
    printf("findScatteringSites: %f\n", time);
}
