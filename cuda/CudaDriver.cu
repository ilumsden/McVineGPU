#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "CudaDriver.hpp"
#include "Kernels.hpp"

/* This is the host-side driver function for setting up the data for the
 * `intersectRectangle` function above, calling said function, and parsing
 * the returned data.
 */
void CudaDriver::handleRectIntersect(const std::shared_ptr<const Box> b, 
                                     const std::vector< std::shared_ptr<Ray> > &rays,
                                     std::vector<float> &host_time)
{
    // N is the number of rays considered
    int N = (int)(rays.size());
    /* Allocates both host and device memory for the float arrays that
     * will be used to store the data passed to the CUDA functions.
     */
    float *rx, *ry, *rz, *vx, *vy, *vz;
    cudaMallocManaged(&rx, N*sizeof(float));
    cudaMallocManaged(&ry, N*sizeof(float));
    cudaMallocManaged(&rz, N*sizeof(float));
    cudaMallocManaged(&vx, N*sizeof(float));
    cudaMallocManaged(&vy, N*sizeof(float));
    cudaMallocManaged(&vz, N*sizeof(float));
    // Copies the data from the rays to the CUDA-compatible arrays.
    int c = 0;
    for (auto ray : rays)
    {
        rx[c] = (float)ray->x;
        ry[c] = (float)ray->y;
        rz[c] = (float)ray->z;
        vx[c] = (float)ray->vx;
        vy[c] = (float)ray->vy;
        vz[c] = (float)ray->vz;
        c++;
    }
    /* Creates a float array in host and device memory to store the 
     * results of the CUDA calculations. The initial value of each element
     * is -5.
     */
    float *device_time;
    cudaMallocManaged(&device_time, 6*N*sizeof(float)+1*sizeof(float));
    for (int i = 0; i < 6*N+1; i++)
    {
        device_time[i] = -5;
    }
    // Calculates the thread and block parameters for CUDA
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    // Starts the CUDA code
    intersectRectangle<<<numBlocks, blockSize>>>(rx, ry, rz, vx, vy, vz, b->x, b->y, b->z, N, device_time);
    // Halts CPU progress until the CUDA code has finished
    cudaDeviceSynchronize();
    // Copies the contents of the device_time array into a vector
    int count = 0;
    while (1)
    {
        if (device_time[count] == -5)
        {
            break;
        }
        else
        {
            host_time.push_back(device_time[count]);
            count++;
        }
    }
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
                 << host_time[i] << "\n";
        }
        else
        {
            std::string buf = "        ";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                 << std::fixed << std::setprecision(5) << std::setw(8) << std::right << host_time[i] << "\n";
        }
    }
    // Closes the file stream
    fout.close();
    // Frees up the memory allocated by the cudaMallocManaged calls above.
    cudaFree(rx);
    cudaFree(ry);
    cudaFree(rz);
    cudaFree(vx);
    cudaFree(vy);
    cudaFree(vz);
    return;
}

/*void CudaDriver::findScatteringSites(const std::shared_ptr<const Box> b,
                                     const std::vector< std::shared_ptr<Ray> > &rays,
                                     const std::vector<float> &int_times, std::vector<float> &sites)
{
    
}*/

// A simple wrapper of the cudaHandler function
void CudaDriver::operator()(std::shared_ptr<Box> b, std::vector< std::shared_ptr<Ray> > &rays)
{
    std::vector<float> int_times;
    handleRectIntersect(b, rays, int_times);
    //std::vector<float> scattering_sites;
    //findScatteringSites(b, rays, int_times, scattering_sites);
}
