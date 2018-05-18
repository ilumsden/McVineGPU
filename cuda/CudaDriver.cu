#include <algorithm>
#include <cfloat>
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
void CudaDriver::handleRectIntersect(std::shared_ptr<Box> &b, 
                                     const std::vector< std::shared_ptr<Ray> > &rays,
                                     std::vector<float> &host_time,
                                     std::vector<float> &int_coords)
{
    // N is the number of rays considered
    int N = (int)(rays.size());
    /* Allocates both host and device memory for the float arrays that
     * will be used to store the data passed to the CUDA functions.
     */
    float *rx, *ry, *rz, *vx, *vy, *vz;
    float *d_rx, *d_ry, *d_rz, *d_vx, *d_vy, *d_vz;
    /*cudaMallocManaged(&rx, N*sizeof(float));
    cudaMallocManaged(&ry, N*sizeof(float));
    cudaMallocManaged(&rz, N*sizeof(float));
    cudaMallocManaged(&vx, N*sizeof(float));
    cudaMallocManaged(&vy, N*sizeof(float));
    cudaMallocManaged(&vz, N*sizeof(float));*/
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
    //cudaMallocManaged(&device_time, 6*N*sizeof(float));
    float *intersect;
    cudaMalloc(&intersect, 6*N*sizeof(float));
    initArray<<<numBlocks, blockSize>>>(intersect, 6*N, FLT_MAX);
    host_time.resize(6*N);
    int_coords.resize(6*N);
    //cudaMallocManaged(&intersect, 6*N*sizeof(float));
    /*for (int i = 0; i < 6*N; i++)
    {
        device_time[i] = -5;
        intersect[i] = FLT_MAX;
    }*/
    // Calculates the thread and block parameters for CUDA
    //int blockSize = 256;
    //int numBlocks = (N + blockSize - 1) / blockSize;
    /*int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(rx, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(ry, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(rz, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(vx, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(vy, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(vz, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(device_time, 6*N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(intersect, 6*N*sizeof(float), device, NULL);*/
    // Starts the CUDA code
    intersectRectangle<<<numBlocks, blockSize>>>(d_rx, d_ry, d_rz, d_vx, d_vy, d_vz, b->x, b->y, b->z, N, device_time, intersect);
    // Halts CPU progress until the CUDA code has finished
    //cudaDeviceSynchronize();
    float *dt = host_time.data();
    float *inters = host_time.data();
    cudaMemcpy(dt, device_time, 6*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(inters, intersect, 6*N*sizeof(float), cudaMemcpyDeviceToHost);
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
    cudaFree(rx);
    cudaFree(ry);
    cudaFree(rz);
    cudaFree(vx);
    cudaFree(vy);
    cudaFree(vz);
    cudaFree(device_time);
    cudaFree(intersect);
    return;
}

void CudaDriver::findScatteringSites(std::shared_ptr< Box> &b,
                                     const std::vector< std::shared_ptr<Ray> > &rays,
                                     const std::vector<float> &int_times, 
                                     const std::vector<float> &int_coords,
                                     std::vector<float> &sites)
{
    int N = (int)(rays.size());
    int tsize = (int)(int_times.size());
    int csize = (int)(int_coords.size());
    float *rx, *ry, *rz, *vx, *vy, *vz, *ts, *inters;
    cudaMallocManaged(&rx, N*sizeof(float));
    cudaMallocManaged(&ry, N*sizeof(float));
    cudaMallocManaged(&rz, N*sizeof(float));
    cudaMallocManaged(&vx, N*sizeof(float));
    cudaMallocManaged(&vy, N*sizeof(float));
    cudaMallocManaged(&vz, N*sizeof(float));
    cudaMallocManaged(&ts, tsize*sizeof(float));
    cudaMallocManaged(&inters, csize*sizeof(float));
    int c = 0;
    for (auto ray : rays)
    {
        rx[c] = (float)ray->x;
        ry[c] = (float)ray->y;
        rz[c] = (float)ray->z;
        vx[c] = (float)ray->vx;
        vy[c] = (float)ray->vy;
        vz[c] = (float)ray->vz;
        for (int i = 0; i < 6; i++)
        {
            ts[6*c + i] = int_times[6*c + i];
            inters[6*c + i] = int_coords[6*c + 1];
        }
        c++;
    }
    float *pos;
    cudaMallocManaged(&pos, 3*N*sizeof(float));
    for (int i = 0; i < 3*N; i++)
    {
        pos[i] = FLT_MAX;
    }
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    curandState *state;
    cudaMallocManaged(&state, blockSize*numBlocks);
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(rx, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(ry, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(rz, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(vx, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(vy, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(vz, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(ts, tsize*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(pos, 3*N*sizeof(float), device, NULL);
    calcScatteringSites<<<numBlocks, blockSize>>>(rx, ry, rz, vx, vy, vz, b->x, b->y, b->z, ts, inters, pos, state, N);
    cudaDeviceSynchronize();
    /*int count = 0;
    while (1)
    {
        if (pos[count] == FLT_MAX)
        {
            break;
        }
        else
        {
            sites.push_back(pos[count]);
            count++;
        }
    }*/
    for (int i = 0; i < 3*N; i++)
    {
        sites.push_back(pos[i]);
    }
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
    cudaFree(rx); 
    cudaFree(ry);
    cudaFree(rz);
    cudaFree(vx); 
    cudaFree(vy);
    cudaFree(vz);
    cudaFree(pos);
    cudaFree(state);
    return;
}

// A simple wrapper of the cudaHandler function
void CudaDriver::operator()(std::shared_ptr<Box> &b, const std::vector< std::shared_ptr<Ray> > &rays)
{
    std::vector<float> int_times;
    std::vector<float> int_coords;
    handleRectIntersect(b, rays, int_times, int_coords);
    std::vector<float> scattering_sites;
    findScatteringSites(b, rays, int_times, int_coords, scattering_sites);
}
