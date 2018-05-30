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

/* This function initializes host- and device-side arrays
 * for storing the initial data for the simulation.
 */
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
    // Copies the data from the rays to the host arrays.
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
    // Copies the data from the host arrays to the device arrays.
    CudaError( cudaMemcpy(d_rx, rx, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaError( cudaMemcpy(d_ry, ry, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaError( cudaMemcpy(d_rz, rz, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaError( cudaMemcpy(d_vx, vx, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaError( cudaMemcpy(d_vy, vy, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaError( cudaMemcpy(d_vz, vz, N*sizeof(float), cudaMemcpyHostToDevice) );
}

/* This destructor deallocates the host- and device-side
 * arrays allocated in the constructor.
 */
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
 * `intersectBlock` function from Kernels.cu, 
 * calling said function, and parsing the returned data.
 */
void CudaDriver::handleRectIntersect(std::shared_ptr<AbstractShape> &b, 
                                     std::vector<float> &host_time,
                                     std::vector<float> &int_coords)
{
    b->intersect(d_rx, d_ry, d_rz, d_vx, d_vy, d_vz, N, blockSize, numBlocks, host_time, int_coords);
    // Opens a file stream and prints the relevant data to time.txt
    // NOTE: this is for debugging purposes only. This will be removed later.
    /*std::fstream fout;
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
    fout.close();*/
    return;
}

/* This function is the host driver function for
 * determining the scattering sites for the neutrons.
 */
void CudaDriver::findScatteringSites(//std::shared_ptr<AbstractShape> &b,
                                     const std::vector<float> &int_times, 
                                     const std::vector<float> &int_coords,
                                     std::vector<float> &sites)
{
    // Stores the sizes of the `int_times` and `int_coords` vectors for later
    int tsize = (int)(int_times.size());
    int csize = (int)(int_coords.size());
    /* Allocates memory for two device-side arrays that store the
     * data passed in from `int_times` and `int_coords`.
     */
    float *ts, *inters;
    CudaError( cudaMalloc(&ts, 2*N*sizeof(float)) );
    CudaError( cudaMalloc(&inters, 6*N*sizeof(float)) );
    CudaError( cudaMemcpy(ts, int_times.data(), 2*N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaError( cudaMemcpy(inters, int_coords.data(), 6*N*sizeof(float), cudaMemcpyHostToDevice) );
    /* `pos` is a device-side array that stores the coordinates of the
     * scattering sites for the neutrons.
     * The default value of its data is FLT_MAX.
     */
    float *pos;
    CudaError( cudaMalloc(&pos, 3*N*sizeof(float)) );
    initArray<<<numBlocks, blockSize>>>(pos, 3*N, FLT_MAX);
    CudaErrorNoCode();
    // Resizes `sites` so that it can store the contents of `pos`.
    sites.resize(3*N);
    /* Allocates an array of curandStates on the device to control
     * the random number generation.
     */
    curandState *state;
    CudaError( cudaMalloc(&state, numBlocks*blockSize*sizeof(curandState)) );
    /* The prepRand function initializes the cuRand random number
     * generator using the states allocated above.
     * NOTE: The chrono-based timing is for debugging purposes.
     *       It will be removed later.
     */
    auto start = std::chrono::steady_clock::now();
    prepRand<<<numBlocks, blockSize>>>(state, time(NULL));
    cudaDeviceSynchronize();
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    printf("Rand Prep Complete\n    Summary: Time = %f\n", time);
    // Calls the kernel for determining the scattering sites for the neutrons
    calcScatteringSites<<<numBlocks, blockSize>>>(ts, inters, pos, state, N);
    CudaErrorNoCode();
    // Copies the post-kernel contents of `pos` into `sites`.
    float* s = sites.data();
    CudaError( cudaMemcpy(s, pos, 3*N*sizeof(float), cudaMemcpyDeviceToHost) );
    // Opens a file stream and prints the 
    // relevant data to scatteringSites.txt
    // NOTE: this is for debugging purposes only. This will be removed later.
    /*std::fstream fout;
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
                 << vx[ind] << " " << vy[ind] << " " << vz[ind] << " || "
                 << int_times[2*ind] << " " << int_times[2*ind+1] << " | " 
                 << sites[i] << "\n";
        }
        else
        {
            std::string buf = "        ";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << "    " << buf << " " << buf << " | "
<< std::fixed << std::setprecision(5) << std::setw(8) << std::right << sites[i] << "\n";
        }
    }
    fout.close();*/
    // Frees the device memory allocated above.
    cudaFree(ts);
    cudaFree(inters);
    cudaFree(pos);
    cudaFree(state);
    return;
}

// This function is the driver for the GPU calculations.
void CudaDriver::runCalculations(std::shared_ptr<AbstractShape> &b)
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
    findScatteringSites(int_times, int_coords, scattering_sites);
    stop = std::chrono::steady_clock::now();
    time = std::chrono::duration<double>(stop - start).count();
    printf("findScatteringSites: %f\n", time);
}
