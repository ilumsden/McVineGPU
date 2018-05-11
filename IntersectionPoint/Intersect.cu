//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <cuda_runtime.h>
#include <device_functions.h>

#include "Intersect.cuh"

__global__ void intersectRectangle(
    float* rx, float* ry, float* rz,
    float* vx, float* vy, float* vz,
    const float X, const float Y, const float Z, const int N,
    float* ts, int &size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    auto calct = [&ts, &size](float x, float y, float z, 
                    float va, float vb, float vc, const float A, const float B)
        {
            float t = (0-z)/vc;
            float r1x = x+va*t; 
            float r1y = y+vb*t;
            if (fabs(r1x) < A/2 && fabs(r1y) < B/2)
            {
                atomicAdd(&size, 1);
                ts[size - 1] = t;
            }
            else
            {
                atomicAdd(&size, 1);
                ts[size - 1] = -1;
            }
        };
    for (int i = index; i < N; i += stride)
    {
        if (vz[i] != 0)
        {
            calct(rx[i], ry[i], rz[i]-Z/2, vx[i], vy[i], vz[i], X, Y);
            calct(rx[i], ry[i], rz[i]+Z/2, vx[i], vy[i], vz[i], X, Y);
        }
        else
        {
            atomicAdd(&size, 1);
            ts[size - 1] = -1;
            atomicAdd(&size, 1);
            ts[size - 1] = -1;
        }
        if (vx[i] != 0)
        {
            calct(ry[i], rz[i], rx[i]-X/2, vy[i], vz[i], vx[i], Y, Z);
            calct(ry[i], rz[i], rx[i]+X/2, vy[i], vz[i], vx[i], Y, Z);
        }
        else
        {
            atomicAdd(&size, 1);
            ts[size - 1] = -1;
            atomicAdd(&size, 1);
            ts[size - 1] = -1;
        }
        if (vy[i] != 0)
        {
            calct(rz[i], rx[i], ry[i]-Y/2, vz[i], vx[i], vy[i], Z, X);
            calct(rz[i], rx[i], ry[i]+Y/2, vz[i], vx[i], vy[i], Z, X);
        }
        else
        {
            atomicAdd(&size, 1);
            ts[size - 1] = -1;
            atomicAdd(&size, 1);
            ts[size - 1] = -1;
        }
    }
}

void CudaDriver::cudaHandler(std::shared_ptr<Box> b, std::vector< std::shared_ptr<Ray> > rays)
{
    int N = (int)(rays.size());
    float *rx, *ry, *rz, *vx, *vy, *vz;
    cudaMallocManaged(&rx, N*sizeof(float));
    cudaMallocManaged(&ry, N*sizeof(float));
    cudaMallocManaged(&rz, N*sizeof(float));
    cudaMallocManaged(&vx, N*sizeof(float));
    cudaMallocManaged(&vy, N*sizeof(float));
    cudaMallocManaged(&vz, N*sizeof(float));
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
    //thrust::host_vector<float> host_time;
    //thrust::device_vector<float> device_time;
    float *device_time;
    cudaMallocManaged(&device_time, 6*N*sizeof(float));
    //device_time.resize(6*N);
    int blockSize = 256;
    int size = 0;
    int numBlocks = (N + blockSize - 1) / blockSize;
    intersectRectangle<<<numBlocks, blockSize>>>(rx, ry, rz, vx, vy, vz, b->x, b->y, b->z, N, device_time, size);
    cudaDeviceSynchronize();
    //thrust::copy(device_time.begin(), device_time.end(), host_time.begin());
    std::vector<float> host_time;
    std::copy(&device_time[0], &device_time[size], host_time.begin());
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
            fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                 << rx[ind] << " " << ry[ind] << " " << rz[ind] << " || "
                 << vx[ind] << " " << vy[ind] << " " << vz[ind] << " | "
                 << host_time[i] << "\n";
        }
        else
        {
            std::string buf = "        ";
            fout << buf << " " << buf << " " << buf << "    " << buf << " " << buf << " " << buf << " | "
                 << std::fixed << std::setprecision(5) << std::setw(8) << std::right << host_time[i] << "\n";
        }
    }
    fout.close();
    cudaFree(rx);
    cudaFree(ry);
    cudaFree(rz);
    cudaFree(vx);
    cudaFree(vy);
    cudaFree(vz);
    return;
}

void CudaDriver::operator()(std::shared_ptr<Box> b, std::vector< std::shared_ptr<Ray> > rays)
{
    cudaHandler(b, rays);
}
