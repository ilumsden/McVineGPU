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
    // Copies the data from the rays to the host arrays.
    int c = 0;
    for (auto ray : rays)
    {
        origins[c] = ray->origin;
        vel[c] = ray->vel;
        c++;
    }
    // Copies the data from the host arrays to the device arrays.
    CudaErrchk( cudaMemcpy(d_origins, origins, N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
    CudaErrchk( cudaMemcpy(d_vel, vel, N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
}

CudaDriver::~CudaDriver()
{
    // Frees the memory for the host-side arrays.
    free(origins);
    free(vel);
    // Frees the memory for the device-side arrays.
    CudaErrchk( cudaFree(d_origins) );
    CudaErrchk( cudaFree(d_vel) );
}

void CudaDriver::handleRectIntersect(std::shared_ptr<AbstractShape> &b, 
                                     std::vector<float> &host_time,
                                     std::vector< Vec3<float> > &int_coords)
{
    /* Calls the shape's intersect function.
     * Inheritance is used to choose the correct algorithm for intersection.
     */
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
        }
    }
    // Closes the file stream
    fout.close();*/
    return;
}

void CudaDriver::findScatteringSites(const std::vector<float> &int_times, 
                                     const std::vector< Vec3<float> > &int_coords,
                                     std::vector< Vec3<float> > &sites)
{
    // Stores the sizes of the `int_times` and `int_coords` vectors for later
    int tsize = (int)(int_times.size());
    int csize = (int)(int_coords.size());
    /* Allocates memory for two device-side arrays that store the
     * data passed in from `int_times` and `int_coords`.
     */
    float *ts;
    Vec3<float> *inters;
    CudaErrchk( cudaMalloc(&ts, 2*N*sizeof(float)) );
    CudaErrchk( cudaMalloc(&inters, 2*N*sizeof(Vec3<float>)) );
    CudaErrchk( cudaMemcpy(ts, int_times.data(), 2*N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaErrchk( cudaMemcpy(inters, int_coords.data(), 2*N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
    /* `pos` is a device-side array that stores the coordinates of the
     * scattering sites for the neutrons.
     * The default value of its data is FLT_MAX.
     */
    Vec3<float> *pos;
    CudaErrchk( cudaMalloc(&pos, N*sizeof(Vec3<float>)) );
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
    Vec3<float>* s = sites.data();
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
            int ind = i;
            fout << "\n";
            fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                 << origins[ind][0] << " " << origins[ind][1] << " " << origins[ind][2] << " || "
                 << vel[ind][0] << " " << vel[ind][1] << " " << vel[ind][2] << " || "
                 << int_times[2*ind] << " " << int_times[2*ind+1] << " | " 
                 << sites[i][0] << "\n";
            std::string buf = "        ";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << "    " << buf << " " << buf << " | "
<< std::fixed << std::setprecision(5) << std::setw(8) << std::right << sites[i][1] << "\n";
            fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << "    " << buf << " " << buf << " | "
<< std::fixed << std::setprecision(5) << std::setw(8) << std::right << sites[i][2] << "\n";
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
    auto start = std::chrono::steady_clock::now();
    handleRectIntersect(b, int_times, int_coords);
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    printf("handleRectIntersect: %f\n", time);
    // Creates the vector that will store the scattering coordinates
    std::vector< Vec3<float> > scattering_sites;
    // Starts the scattering site calculation
    start = std::chrono::steady_clock::now();
    findScatteringSites(int_times, int_coords, scattering_sites);
    stop = std::chrono::steady_clock::now();
    time = std::chrono::duration<double>(stop - start).count();
    printf("findScatteringSites: %f\n", time);
}
