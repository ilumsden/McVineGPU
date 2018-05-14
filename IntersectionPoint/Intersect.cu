#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <fstream>

#include "Intersect.cuh"

__device__ int size_shared;

__device__ void calculate_time(float* ts, float x, float y, float z,
                               float va, float vb, float vc, const float A, const float B)
{
    float t = (0-z)/vc;
    float r1x = x+va*t; 
    float r1y = y+vb*t;
    if (fabs(r1x) < A/2 && fabs(r1y) < B/2)
    {
        int size = atomicAdd(&size_shared, 1);
        ts[size - 1] = t;
    }
    else
    {
        int size = atomicAdd(&size_shared, 1);
        ts[size - 1] = -1;
    }
}

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
            calculate_time(ts, rx[index], ry[index], rz[index]-Z/2, vx[index], vy[index], vz[index], X, Y);
            calculate_time(ts, rx[index], ry[index], rz[index]+Z/2, vx[index], vy[index], vz[index], X, Y);
        }
        else
        {
            int size = atomicAdd(&size_shared, 1);
            ts[size - 1] = -1;
            size = atomicAdd(&size_shared, 1);
            ts[size - 1] = -1;
        }
        if (vx[index] != 0)
        {
            calculate_time(ts, ry[index], rz[index], rx[index]-X/2, vy[index], vz[index], vx[index], Y, Z);
            calculate_time(ts, ry[index], rz[index], rx[index]+X/2, vy[index], vz[index], vx[index], Y, Z);
        }
        else
        {
            int size = atomicAdd(&size_shared, 1);
            ts[size - 1] = -1;
            size = atomicAdd(&size_shared, 1);
            ts[size - 1] = -1;
        }
        if (vy[index] != 0)
        {
            calculate_time(ts, rz[index], rx[index], ry[index]-Y/2, vz[index], vx[index], vy[index], Z, X);
            calculate_time(ts, rz[index], rx[index], ry[index]+Y/2, vz[index], vx[index], vy[index], Z, X);
        }
        else
        {
            int size = atomicAdd(&size_shared, 1);
            ts[size - 1] = -1;
            size = atomicAdd(&size_shared, 1);
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
    float *device_time;
    cudaMallocManaged(&device_time, 6*N*sizeof(float)+1*sizeof(float));
    for (int i = 0; i < 6*N+1; i++)
    {
        device_time[i] = -5;
    }
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    intersectRectangle<<<numBlocks, blockSize>>>(rx, ry, rz, vx, vy, vz, b->x, b->y, b->z, N, device_time);
    cudaDeviceSynchronize();
    printf("\n");
    for (int i = 0; i < 6*N; i++)
    {
        printf("t = %f\n", device_time[i]);
        printf("i = %i\n\n", i);
    }
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
