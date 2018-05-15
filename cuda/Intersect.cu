#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "Intersect.hpp"

/* This is a device-only helper function for determining the time
 * it takes a ray to intersect the rectangle specified by the `intersectRectangle`
 * function.
 * It is a CUDA version of the intersectRectangle function from ArrowIntersector.cc
 * in McVine (mcvine/packages/mccomposite/lib/geometry/visitors/ArrowIntersector.cc).
 */
__device__ void calculate_time(float* ts, float x, float y, float z,
                               float va, float vb, float vc, const float A, const float B, const int offset)
{
    __syncthreads();
    float t = (0-z)/vc;
    float r1x = x+va*t; 
    float r1y = y+vb*t;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (fabs(r1x) < A/2 && fabs(r1y) < B/2)
    {
        ts[offset + index*6] = t;
    }
    else
    {
        ts[offset + index*6] = -1;
    }
}

/* This is a global CUDA function for controlling the calculation of intersection
 * times. It is a CUDA version of the visit function from ArrowIntersector.cc in
 * McVine (mcvine/packages/mccomposite/lib/geometry/visitors/ArrowIntersector.cc).
 */
__global__ void intersectRectangle(
    float* rx, float* ry, float* rz,
    float* vx, float* vy, float* vz,
    const float X, const float Y, const float Z, const int N,
    float* ts)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        if (vz[index] != 0)
        {
            calculate_time(ts, rx[index], ry[index], rz[index]-Z/2, vx[index], vy[index], vz[index], X, Y, 0);
            calculate_time(ts, rx[index], ry[index], rz[index]+Z/2, vx[index], vy[index], vz[index], X, Y, 1);
        }
        else
        {
            ts[index*6] = -1;
            ts[index*6 + 1] = -1;
        }
        if (vx[index] != 0)
        {
            calculate_time(ts, ry[index], rz[index], rx[index]-X/2, vy[index], vz[index], vx[index], Y, Z, 2);
            calculate_time(ts, ry[index], rz[index], rx[index]+X/2, vy[index], vz[index], vx[index], Y, Z, 3);
        }
        else
        {
            ts[index*6 + 2] = -1;
            ts[index*6 + 3] = -1;
        }
        if (vy[index] != 0)
        {
            calculate_time(ts, rz[index], rx[index], ry[index]-Y/2, vz[index], vx[index], vy[index], Z, X, 4);
            calculate_time(ts, rz[index], rx[index], ry[index]+Y/2, vz[index], vx[index], vy[index], Z, X, 5);
        }
        else
        {
            ts[index*6 + 4] = -1;
            ts[index*6 + 5] = -1;
        }
    }
}

/* This is the host-side driver function for setting up the data for the
 * `intersectRectangle` function above, calling said function, and parsing
 * the returned data.
 */
void CudaDriver::cudaHandler(std::shared_ptr<Box> b, std::vector< std::shared_ptr<Ray> > &rays)
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
    /*printf("\n");
    for (int i = 0; i < 6*N; i++)
    {
        printf("t = %f\n", device_time[i]);
    }*/
    // Copies the contents of the device_time array into a vector
    std::vector<float> host_time;
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

// A simple wrapper of the cudaHandler function
void CudaDriver::operator()(std::shared_ptr<Box> b, std::vector< std::shared_ptr<Ray> > &rays)
{
    cudaHandler(b, rays);
}
