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
    N = (int)(rays.size());
    // Calculates the CUDA launch parameters using bS
    blockSize = bS;
    numBlocks = (N + blockSize - 1) / blockSize;
    printf("blockSize = %i\nnumBlocks = %i\n", blockSize, numBlocks);
    /* Allocates both host and device memory for the float arrays that
     * will be used to store the data passed to the CUDA functions.
     */
    origins = (Vec3<float>*)malloc(N*sizeof(Vec3<float>));
    CudaErrchk( cudaMalloc(&d_origins, N*sizeof(Vec3<float>)) );
    vel = (Vec3<float>*)malloc(N*sizeof(Vec3<float>));
    CudaErrchk( cudaMalloc(&d_vel, N*sizeof(Vec3<float>)) );
    /*rx = (float*)malloc(N*sizeof(float));
    CudaErrchk( cudaMalloc(&d_rx, N*sizeof(float)) );
    ry = (float*)malloc(N*sizeof(float));
    CudaErrchk( cudaMalloc(&d_ry, N*sizeof(float)) );
    rz = (float*)malloc(N*sizeof(float));
    CudaErrchk( cudaMalloc(&d_rz, N*sizeof(float)) );
    vx = (float*)malloc(N*sizeof(float));
    CudaErrchk( cudaMalloc(&d_vx, N*sizeof(float)) );
    vy = (float*)malloc(N*sizeof(float));
    CudaErrchk( cudaMalloc(&d_vy, N*sizeof(float)) );
    vz = (float*)malloc(N*sizeof(float));
    CudaErrchk( cudaMalloc(&d_vz, N*sizeof(float)) );*/
    // Copies the data from the rays to the host arrays.
    int c = 0;
    for (auto ray : rays)
    {
        origins[c] = ray->origin;
        vel[c] = ray->vel;
        /*rx[c] = (float)(ray->x);
        ry[c] = (float)(ray->y);
        rz[c] = (float)(ray->z);
        vx[c] = (float)(ray->vx);
        vy[c] = (float)(ray->vy);
        vz[c] = (float)(ray->vz);*/
        c++;
    }
    // Copies the data from the host arrays to the device arrays.
    CudaErrchk( cudaMemcpy(d_origins, origins, N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
    CudaErrchk( cudaMemcpy(d_vel, vel, N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
    /*CudaErrchk( cudaMemcpy(d_rx, rx, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaErrchk( cudaMemcpy(d_ry, ry, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaErrchk( cudaMemcpy(d_rz, rz, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaErrchk( cudaMemcpy(d_vx, vx, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaErrchk( cudaMemcpy(d_vy, vy, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaErrchk( cudaMemcpy(d_vz, vz, N*sizeof(float), cudaMemcpyHostToDevice) );*/
}

CudaDriver::~CudaDriver()
{
    // Frees the memory for the host-side arrays.
    free(origins);
    free(vel);
    /*free(rx);
    free(ry);
    free(rz);
    free(vx);
    free(vy);
    free(vz);*/
    // Frees the memory for the device-side arrays.
    CudaErrchk( cudaFree(d_origins) );
    CudaErrchk( cudaFree(d_vel) );
    /*CudaErrchk( cudaFree(d_rx) );
    CudaErrchk( cudaFree(d_ry) );
    CudaErrchk( cudaFree(d_rz) );
    CudaErrchk( cudaFree(d_vx) );
    CudaErrchk( cudaFree(d_vy) );
    CudaErrchk( cudaFree(d_vz) );*/
}

void CudaDriver::handleRectIntersect(std::shared_ptr<AbstractShape> &b, 
                                     std::vector<float> &host_time,
                                     std::vector< Vec3<float> > &int_coords)
                                     //std::vector<float> &int_coords)
{
    /* Calls the shape's intersect function.
     * Inheritance is used to choose the correct algorithm for intersection.
     */
    //b->intersect(d_rx, d_ry, d_rz, d_vx, d_vy, d_vz, N, blockSize, numBlocks, host_time, int_coords);
    b->intersect(d_origins, d_vel, N, blockSize, numBlocks, host_time, int_coords);
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
        //printf("print i = %i\n", i);
        std::string buf = "        ";
        if (i % 2 == 0)
        {
            int ind = i/2;
            fout << "\n";
            fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                 << origins[ind][0] << " " << origins[ind][1] << " " << origins[ind][2] << " || "
                 << vel[ind][0] << " " << vel[ind][1] << " " << vel[ind][2] << " | "
                 << host_time[i] << " / " << int_coords[i][0] << "\n";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                 << std::fixed << std::setprecision(5) << std::setw(8) << std::right << buf << " / " << int_coords[i][1] << "\n";
            std::string buf = "        ";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                 << std::fixed << std::setprecision(5) << std::setw(8) << std::right << buf << " / " << int_coords[i][2] << "\n";
        }
        else
        {
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                 << std::fixed << std::setprecision(5) << std::setw(8) << std::right << host_time[i] << " / " << int_coords[i][0] << "\n";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                 << std::fixed << std::setprecision(5) << std::setw(8) << std::right << buf << " / " << int_coords[i][1] << "\n";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                 << std::fixed << std::setprecision(5) << std::setw(8) << std::right << buf << " / " << int_coords[i][2] << "\n";
        }*/
        /*if (i % 6 == 0)
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
        }*/
    /*}
    // Closes the file stream
    fout.close();*/
    return;
}

void CudaDriver::findScatteringSites(const std::vector<float> &int_times, 
                                     //const std::vector<float> &int_coords,
                                     //std::vector<float> &sites)
                                     const std::vector< Vec3<float> > &int_coords,
                                     std::vector< Vec3<float> > &sites)
{
    // Stores the sizes of the `int_times` and `int_coords` vectors for later
    int tsize = (int)(int_times.size());
    int csize = (int)(int_coords.size());
    /* Allocates memory for two device-side arrays that store the
     * data passed in from `int_times` and `int_coords`.
     */
    float *ts;//, *inters;
    Vec3<float> *inters;
    CudaErrchk( cudaMalloc(&ts, 2*N*sizeof(float)) );
    //CudaErrchk( cudaMalloc(&inters, 6*N*sizeof(float)) );
    CudaErrchk( cudaMalloc(&inters, 2*N*sizeof(Vec3<float>)) );
    CudaErrchk( cudaMemcpy(ts, int_times.data(), 2*N*sizeof(float), cudaMemcpyHostToDevice) );
    //CudaErrchk( cudaMemcpy(inters, int_coords.data(), 6*N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaErrchk( cudaMemcpy(inters, int_coords.data(), 2*N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
    /* `pos` is a device-side array that stores the coordinates of the
     * scattering sites for the neutrons.
     * The default value of its data is FLT_MAX.
     */
    //float *pos;
    Vec3<float> *pos;
    //CudaErrchk( cudaMalloc(&pos, 3*N*sizeof(float)) );
    CudaErrchk( cudaMalloc(&pos, N*sizeof(Vec3<float>)) );
    //initArray<<<numBlocks, blockSize>>>(pos, 3*N, FLT_MAX);
    initArray< Vec3<float> ><<<numBlocks, blockSize>>>(pos, N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
    CudaErrchkNoCode();
    // Resizes `sites` so that it can store the contents of `pos`.
    //sites.resize(3*N);
    sites.resize(N);
    /* Allocates an array of curandStates on the device to control
     * the random number generation.
     */
    curandState *state;
    CudaErrchk( cudaMalloc(&state, numBlocks*blockSize*sizeof(curandState)) );
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
    CudaErrchkNoCode();
    // Copies the post-kernel contents of `pos` into `sites`.
    //float* s = sites.data();
    Vec3<float>* s = sites.data();
    //CudaErrchk( cudaMemcpy(s, pos, 3*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(s, pos, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    // Opens a file stream and prints the 
    // relevant data to scatteringSites.txt
    // NOTE: this is for debugging purposes only. This will be removed later.
    std::fstream fout;
    fout.open("scatteringSites.txt", std::ios::out);
    if (!fout.is_open())
    {
        std::cerr << "scatteringSites.txt could not be opened.\n";
        exit(-2);
    }
    for (int i = 0; i < (int)(sites.size()); i++)
    {
        //if (i % 3 == 0)
        //{
            int ind = i;//i/3;
            fout << "\n";
            fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                 //<< rx[ind] << " " << ry[ind] << " " << rz[ind] << " || "
                 << origins[ind][0] << " " << origins[ind][1] << " " << origins[ind][2] << " || "
                 //<< vx[ind] << " " << vy[ind] << " " << vz[ind] << " || "
                 << vel[ind][0] << " " << vel[ind][1] << " " << vel[ind][2] << " || "
                 << int_times[2*ind] << " " << int_times[2*ind+1] << " | " 
                 << sites[i][0] << "\n";
        //}
        //else
        //{
            std::string buf = "        ";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << "    " << buf << " " << buf << " | "
<< std::fixed << std::setprecision(5) << std::setw(8) << std::right << sites[i][1] << "\n";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << "    " << buf << " " << buf << " | "
<< std::fixed << std::setprecision(5) << std::setw(8) << std::right << sites[i][2] << "\n";
        //}
    }
    fout.close();
    // Frees the device memory allocated above.
    cudaFree(ts);
    cudaFree(inters);
    cudaFree(pos);
    cudaFree(state);
    return;
}

void CudaDriver::runCalculations(std::shared_ptr<AbstractShape> &b)
{
    /* Creates the vectors that will store the intersection
     * times and coordinates.
     */
    std::vector<float> int_times;
    std::vector< Vec3<float> > int_coords;
    //std::vector<float> int_coords;
    // Starts the intersection calculation
    auto start = std::chrono::steady_clock::now();
    handleRectIntersect(b, int_times, int_coords);
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    printf("handleRectIntersect: %f\n", time);
    // Creates the vector that will store the scattering coordinates
    //std::vector<float> scattering_sites;
    std::vector< Vec3<float> > scattering_sites;
    // Starts the scattering site calculation
    start = std::chrono::steady_clock::now();
    findScatteringSites(int_times, int_coords, scattering_sites);
    stop = std::chrono::steady_clock::now();
    time = std::chrono::duration<double>(stop - start).count();
    printf("findScatteringSites: %f\n", time);
}
