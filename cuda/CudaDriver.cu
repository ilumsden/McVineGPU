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
#include "ScatteringKernels.hpp"
#include "Error.hpp"

CudaDriver::CudaDriver(std::vector< std::shared_ptr<Ray> > &rays, int bS)
{ 
    N = (int)(rays.size());
    // Calculates the CUDA launch parameters using bS
    blockSize = bS;
    numBlocks = (N + blockSize - 1) / blockSize;
    printf("blockSize = %i\nnumBlocks = %i\n", blockSize, numBlocks);
    /* Allocates both host and device memory for the float arrays that
     * will be used to store the data passed to the CUDA functions.
     */
    rayptr = &rays;
    origins = (Vec3<float>*)malloc(N*sizeof(Vec3<float>));
    CudaErrchk( cudaMalloc(&d_origins, N*sizeof(Vec3<float>)) );
    vel = (Vec3<float>*)malloc(N*sizeof(Vec3<float>));
    CudaErrchk( cudaMalloc(&d_vel, N*sizeof(Vec3<float>)) );
    times = (float*)malloc(N*sizeof(float));
    CudaErrchk( cudaMalloc(&d_times, N*sizeof(float)) );
    probs = (float*)malloc(N*sizeof(float));
    CudaErrchk( cudaMalloc(&d_probs, N*sizeof(float)) );
    // Copies the data from the rays to the host arrays.
    int c = 0;
    for (auto ray : rays)
    {
        origins[c] = ray->origin;
        vel[c] = ray->vel;
        times[c] = ray->t;
        probs[c] = ray->prob;
        c++;
    }
    // Copies the data from the host arrays to the device arrays.
    CudaErrchk( cudaMemcpy(d_origins, origins, N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
    CudaErrchk( cudaMemcpy(d_vel, vel, N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
    CudaErrchk( cudaMemcpy(d_times, times, N*sizeof(float), cudaMemcpyHostToDevice) );
    CudaErrchk( cudaMemcpy(d_probs, probs, N*sizeof(float), cudaMemcpyHostToDevice) );
}

CudaDriver::~CudaDriver()
{
    // Frees the memory for the host-side arrays.
    free(origins);
    free(vel);
    free(times);
    free(probs);
    // Frees the memory for the device-side arrays.
    CudaErrchk( cudaFree(d_origins) );
    CudaErrchk( cudaFree(d_vel) );
    CudaErrchk( cudaFree(d_times) );
    CudaErrchk( cudaFree(d_probs) );
}

void CudaDriver::printData(const std::string &fname)
{
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::fstream fout;
    if (fname != std::string())
    {
        fout.open(fname.c_str(), std::ios::out);
        if (!fout.is_open())
        {
            std::cerr << fname << " cannot be openned.\n";
            exit(-2);
        }
        std::cout.rdbuf(fout.rdbuf());
    }
    std::string buf = "        ";
    std::cout << "Position" << " " << buf << " " << buf << " || "
              << "Velocity" << " " << buf << " " << buf << " || "
              << "  Time  " << " || " << "Probability" << "\n\n";
    for (int i = 0; i < N; i++)
    {
        std::cout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
	         << origins[i][0]
		 << " " << origins[i][1]
		 << " " << origins[i][2]
		 << " || "
                 << vel[i][0]
                 << " " << vel[i][1]
                 << " " << vel[i][2]
                 << " || "
                 << times[i]
                 << " || "
                 << probs[i]
                 << "\n";
    }
    std::cout.rdbuf(coutbuf);
    if (fname != std::string())
    {
        fout.close();
    }
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
                                     const std::vector< Vec3<float> > &int_coords)
                                     //std::vector< Vec3<float> > &sites)
{
    // Uncomment with printing
    //std::vector< Vec3<float> > tmp;
    //tmp.resize(N);
    //Vec3<float> *ta = tmp.data();
    //memcpy(ta, origins, N*sizeof(Vec3<float>));
    // Stores the sizes of the `int_times` and `int_coords` vectors for later
    int tsize = (int)(int_times.size());
    /* Allocates memory for two device-side arrays that store the
     * data passed in from `int_times` and `int_coords`.
     */
    float *ts;
    CudaErrchk( cudaMalloc(&ts, 2*N*sizeof(float)) );
    CudaErrchk( cudaMemcpy(ts, int_times.data(), 2*N*sizeof(float), cudaMemcpyHostToDevice) );
    /* `pos` is a device-side array that stores the coordinates of the
     * scattering sites for the neutrons.
     * The default value of its data is FLT_MAX.
     */
    Vec3<float> *pos;
    CudaErrchk( cudaMalloc(&pos, N*sizeof(Vec3<float>)) );
    initArray< Vec3<float> ><<<numBlocks, blockSize>>>(pos, N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
    CudaErrchkNoCode();
    float *scat_times;
    CudaErrchk( cudaMalloc(&scat_times, N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(scat_times, N, -5);
    // Resizes `sites` so that it can store the contents of `pos`.
    //sites.resize(N);
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
    //calcScatteringSites<<<numBlocks, blockSize>>>(ts, inters, pos, state, N);
    calcScatteringSites<<<numBlocks, blockSize>>>(ts, d_origins, d_vel, pos, scat_times, state, N);
    propagate<<<numBlocks, blockSize>>>(d_origins, d_times, pos, scat_times, N);
    CudaErrchkNoCode();
    Vec3<float> *ic;
    CudaErrchk( cudaMalloc(&ic, 2*N*sizeof(Vec3<float>)) );
    CudaErrchk( cudaMemcpy(ic, int_coords.data(), 2*N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
    printf("attenuation = %f\n", atten);
    updateProbability<<<numBlocks, blockSize>>>(d_probs, d_origins, ic, atten, N);
    CudaErrchkNoCode();
    CudaErrchk( cudaMemcpy(origins, d_origins, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(times, d_times, N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(probs, d_probs, N*sizeof(float), cudaMemcpyDeviceToHost) );
    /*std::fstream fout;
    fout.open("scatteringSites.txt", std::ios::out);
    if (!fout.is_open())
    {
        std::cerr << "scatteringSites.txt could not be opened.\n";
        exit(-2);
    }
    for (int i = 0; i < N; i++)
    {
        int ind = 2*i;
        fout << "\n";
        fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
             << tmp[i][0] << " " << tmp[i][1] << " " << tmp[i][2] << " || "
             << vel[i][0] << " " << vel[i][1] << " " << vel[i][2] << " || "
             << int_times[ind] << " " << int_times[ind+1] << " | "
             << origins[i][0] << "\n";
        std::string buf = "        ";
        fout << buf << " " << buf << " " << buf << "    "
             << buf << " " << buf << " " << buf << "    "
             << buf << " " << buf << "   "
             << std::fixed << std::setprecision(5) << std::setw(8) << std::right
             << origins[i][1] << "\n";
        fout << buf << " " << buf << " " << buf << "    "
             << buf << " " << buf << " " << buf << "    "
             << buf << " " << buf << "   "
             << std::fixed << std::setprecision(5) << std::setw(8) << std::right
             << origins[i][2] << "\n";
        fout << buf << " " << buf << " " << buf << "    "
             << buf << " " << buf << " " << buf << "    "
             << buf << " " << buf << "   "
             << std::fixed << std::setprecision(5) << std::setw(8) << std::right
             << times[i] << "\n";
    }
    fout.close();*/
    // Frees the device memory allocated above.
    CudaErrchk( cudaFree(ts) );
    CudaErrchk( cudaFree(ic) );
    CudaErrchk( cudaFree(pos) );
    CudaErrchk( cudaFree(state) );
    return;
}

void CudaDriver::findScatteringVels()//const std::vector<float> &int_times)//,
                                    //std::vector< Vec3<float> > &scattering_vels)
{
    /*Vec3<float> *d_postVel;
    CudaErrchk( cudaMalloc(&d_postVel, N*sizeof(Vec3<float>)) );
    initArray< Vec3<float> ><<<numBlocks, blockSize>>>(d_postVel, N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));*/
    /*float *d_times;
    CudaErrchk( cudaMalloc(&d_times, 2*N*sizeof(float)) );
    CudaErrchk( cudaMemcpy(d_times, int_times.data(), 2*N*sizeof(float), cudaMemcpyHostToDevice) );*/
    //scattering_vels.resize(N);
    //CudaErrchkNoCode();
    // Uncomment during printing
    //std::vector< Vec3<float> > tmp;
    //tmp.resize(N);
    //Vec3<float> *ta = tmp.data();
    //memcpy(ta, vel, N*sizeof(Vec3<float>));
    curandState *state;
    CudaErrchk( cudaMalloc(&state, numBlocks*blockSize*sizeof(curandState)) );
    auto start = std::chrono::steady_clock::now();
    prepRand<<<numBlocks, blockSize>>>(state, time(NULL));
    CudaErrchkNoCode();
    CudaErrchk( cudaDeviceSynchronize() );
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    printf("Rand Prep Complete\n    Summary: Time = %f\n", time);
    elasticScatteringKernel<<<numBlocks, blockSize>>>(d_times,
                                                      d_vel,
                                                      state, N);
    CudaErrchkNoCode();
    CudaErrchk( cudaMemcpy(vel, d_vel, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    //Vec3<float> *sv = scattering_vels.data();
    //CudaErrchk( cudaMemcpy(sv, d_postVel, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    // Opens a file stream and prints the 
    // relevant data to scatteringVels.txt
    // NOTE: this is for debugging purposes only. This will be removed later.
    /*std::fstream fout;
    fout.open("scatteringVels.txt", std::ios::out);
    if (!fout.is_open())
    {
        std::cerr << "scatteringVels.txt could not be opened.\n";
        exit(-2);
    }
    for (int i = 0; i < N; i++)
    {
        fout << "\n";
        fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
             << tmp[i][0] << " " << tmp[i][1] << " " << tmp[i][2] << " || "
             << vel[i][0] << " " << vel[i][1] << " " << vel[i][2] << "\n";
    }
    fout.close();*/
    //CudaErrchk( cudaFree(d_postVel) );
    //CudaErrchk( cudaFree(d_times) );
    CudaErrchk( cudaFree(state) );
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
    //std::vector< Vec3<float> > scattering_sites;
    // Starts the scattering site calculation
    start = std::chrono::steady_clock::now();
    findScatteringSites(int_times, int_coords);//scattering_sites);
    stop = std::chrono::steady_clock::now();
    time = std::chrono::duration<double>(stop - start).count();
    printf("findScatteringSites: %f\n", time);
    //std::vector< Vec3<float> > scattering_vels;
    start = std::chrono::steady_clock::now();
    findScatteringVels();//int_times, scattering_vels);
    stop = std::chrono::steady_clock::now();
    time = std::chrono::duration<double>(stop - start).count();
    printf("findScatteringTimes: %f\n", time);
}
