#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <chrono>

#include "CudaDriver.hpp"
#include "Kernels.hpp"

void CudaDriver::allocInitialData(const std::vector< std::shared_ptr<Ray> > &rays)
{ 
    // N is the number of rays considered
    N = (int)(rays.size());
    /* Allocates both host and device memory for the float arrays that
     * will be used to store the data passed to the CUDA functions.
     */
    rx = (float*)malloc(N*sizeof(float));
    cudaMalloc(&d_rx, N*sizeof(float));
    ry = (float*)malloc(N*sizeof(float));
    cudaMalloc(&d_ry, N*sizeof(float));
    rz = (float*)malloc(N*sizeof(float));
    cudaMalloc(&d_rz, N*sizeof(float));
    vx = (float*)malloc(N*sizeof(float));
    cudaMalloc(&d_vx, N*sizeof(float));
    vy = (float*)malloc(N*sizeof(float));
    cudaMalloc(&d_vy, N*sizeof(float));
    vz = (float*)malloc(N*sizeof(float));
    cudaMalloc(&d_vz, N*sizeof(float));
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
    cudaMemcpy(d_rx, rx, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ry, ry, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rz, rz, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, vx, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, vy, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, vz, N*sizeof(float), cudaMemcpyHostToDevice);
}

void CudaDriver::freeInitialData()
{
    free(rx);
    free(ry);
    free(rz);
    free(vx);
    free(vy);
    free(vz);
    cudaFree(d_rx);
    cudaFree(d_ry);
    cudaFree(d_rz);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
}

/* This is the host-side driver function for setting up the data for the
 * `intersectRectangle` function above, calling said function, and parsing
 * the returned data.
 */
void CudaDriver::handleRectIntersect(std::shared_ptr<Box> &b, 
                                     std::vector<float> &host_time,
                                     std::vector<float> &int_coords)
{
    // Calculates the thread and block parameters for CUDA
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    /* Creates a float array in host and device memory to store the 
     * results of the CUDA calculations. The initial value of each element
     * is -5.
     */
    float *device_time;
    cudaMalloc(&device_time, 6*N*sizeof(float));
    initArray<<<numBlocks, blockSize>>>(device_time, 6*N, -5);
    float *intersect;
    cudaMalloc(&intersect, 6*N*sizeof(float));
    initArray<<<numBlocks, blockSize>>>(intersect, 6*N, FLT_MAX);
    host_time.resize(6*N);
    int_coords.resize(6*N);
    // Starts the CUDA code
    intersectRectangle<<<numBlocks, blockSize>>>(d_rx, d_ry, d_rz, d_vx, d_vy, d_vz, b->x, b->y, b->z, N, device_time, intersect);
    // Halts CPU progress until the CUDA code has finished
    cudaMemcpy(host_time.data(), device_time, 6*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(int_coords.data(), intersect, 6*N*sizeof(float), cudaMemcpyDeviceToHost);
    // Opens a file stream and prints the relevant data to time.txt
    // NOTE: this is for debugging purposes only. This will be removed later.
    std::fstream fout;
    fout.open("time.txt", std::ios::out);
    if (!fout.is_open())
    {
        std::cerr << "time.txt could not be opened.\n";
        exit(-1);
    }
    for (int i = 0; i < (int)(host_time.size()); i++)
    {
        if (i % 6 == 0)
        {
            int ind = i/6;
            fout << "\n";
            fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                 << rx[ind] << " " << ry[ind] << " " << rz[ind] << " || "
                 << vx[ind] << " " << vy[ind] << " " << vz[ind] << " | "
                 << host_time[i] << " / " << int_coords[i] << "\n";
        }
        else
        {
            std::string buf = "        ";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                 << std::fixed << std::setprecision(5) << std::setw(8) << std::right << host_time[i] << " / " << int_coords[i] << "\n";
        }
    }
    // Closes the file stream
    fout.close();
    // Frees up the memory allocated by the cudaMallocManaged calls above.
    cudaFree(device_time);
    cudaFree(intersect);
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
    cudaMalloc(&ts, 6*N*sizeof(float));
    cudaMalloc(&inters, 6*N*sizeof(float));
    cudaMemcpy(ts, int_times.data(), 6*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inters, int_coords.data(), 6*N*sizeof(float), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    float *pos;
    cudaMalloc(&pos, 3*N*sizeof(float));
    initArray<<<numBlocks, blockSize>>>(pos, 3*N, FLT_MAX);
    sites.resize(3*N);
    curandState *state;
    cudaMalloc(&state, blockSize*numBlocks);
    calcScatteringSites<<<numBlocks, blockSize>>>(d_rx, d_ry, d_rz, d_vx, d_vy, d_vz, b->x, b->y, b->z, ts, inters, pos, state, N);
    cudaMemcpy(sites.data(), pos, 3*N*sizeof(float), cudaMemcpyDeviceToHost);
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
void CudaDriver::operator()(std::shared_ptr<Box> &b, const std::vector< std::shared_ptr<Ray> > &rays)
{
    allocInitialData(rays);
    std::vector<float> int_times;
    std::vector<float> int_coords;
    handleRectIntersect(b, int_times, int_coords);
    std::vector<float> scattering_sites;
    findScatteringSites(b, int_times, int_coords, scattering_sites);
    freeInitialData();
}
