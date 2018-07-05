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
    /* If there is a file name provided (i.e. fname != str::string()),
     * the C++ stdout stream (cout) is redirected to print to the
     * desired file. Otherwise, all data is printed to stdout.
     */
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
    // A generic buffer for separation purposes
    std::string buf = "        ";
    // Prints header info
    std::cout << "Position" << " " << buf << " " << buf << " || "
              << "Velocity" << " " << buf << " " << buf << " || "
              << "  Time  " << " || " << "Probability" << "\n\n";
    // Prints the data for each neutron
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
                 << "\n\n";
    }
    /* If cout was redirected, this "fixes" it so that it prints to
     * stdout in the future. Otherwise, this does nothing.
     */
    std::cout.rdbuf(coutbuf);
    // Closes the file stream if it was ever openned.
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
{
    // Uncomment with printing
    /*std::vector< Vec3<float> > tmp;
    tmp.resize(N);
    Vec3<float> *ta = tmp.data();
    memcpy(ta, origins, N*sizeof(Vec3<float>));*/
    // Stores the size of the `int_times` for later
    int tsize = (int)(int_times.size());
    /* Allocates memory for a device-side array that stores the
     * data passed in from `int_times`.
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
    /* `scat_times` is a device_side array that stores the times at
     * which the neutrons reach their scattering sites.
     * The default value of its data is -5.
     */
    float *scat_times;
    CudaErrchk( cudaMalloc(&scat_times, N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(scat_times, N, -5);
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
    calcScatteringSites<<<numBlocks, blockSize>>>(ts, d_origins, d_vel, pos, scat_times, state, N);
    /* Propagates the neutrons to their scattering sites.
     * In other words, the scattering coordinates and times are copied
     * into the device arrays that store the neutrons' origins and times
     * (d_origins and d_times respectively).
     */
    propagate<<<numBlocks, blockSize>>>(d_origins, d_times, pos, scat_times, N);
    CudaErrchkNoCode();
    /* `ic` is a device-side array that stores the intersection
     * coordinates between the neutron and scattering body, as calculated
     * in the handleIntersect function.
     */
    Vec3<float> *ic;
    CudaErrchk( cudaMalloc(&ic, 2*N*sizeof(Vec3<float>)) );
    CudaErrchk( cudaMemcpy(ic, int_coords.data(), 2*N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
    /* Updates the probability attribute of the neutrons to account
     * for the absorption that occurs as a neutron travels through the
     * scattering body to the scattering site.
     */
    updateProbability<<<numBlocks, blockSize>>>(d_probs, d_origins, ic, atten, N);
    CudaErrchkNoCode();
    // Updates the host-side arrays for the edited neutron data.
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

void CudaDriver::findScatteringVels()
{
    // Uncomment during printing
    /*std::vector< Vec3<float> > tmp;
    tmp.resize(N);
    Vec3<float> *ta = tmp.data();
    memcpy(ta, vel, N*sizeof(Vec3<float>));*/
    /* Allocates an array of curandStates on the device to control
     * the random number generation.
     */
    curandState *state;
    CudaErrchk( cudaMalloc(&state, numBlocks*blockSize*sizeof(curandState)) );
    auto start = std::chrono::steady_clock::now();
    prepRand<<<numBlocks, blockSize>>>(state, time(NULL));
    CudaErrchkNoCode();
    CudaErrchk( cudaDeviceSynchronize() );
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    printf("Rand Prep Complete\n    Summary: Time = %f\n", time);
    /* Calls the elasticScatteringKernel function to update the neutron
     * velocities post-elastic scattering.
     */
    elasticScatteringKernel<<<numBlocks, blockSize>>>(d_times,
                                                      d_vel,
                                                      state, N);
    CudaErrchkNoCode();
    /* Copies the new neutron velocities into the host-side neutron
     * velocity array.
     */
    CudaErrchk( cudaMemcpy(vel, d_vel, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
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
    CudaErrchk( cudaFree(state) );
}

void CudaDriver::runCalculations(std::shared_ptr<AbstractShape> &b)
{
    /* Creates the vectors that will store the intersection
     * times and coordinates.
     */
    std::vector<float> int_times;
    std::vector< Vec3<float> > int_coords;
    // Starts the intersection calculation
    auto start = std::chrono::steady_clock::now();
    handleRectIntersect(b, int_times, int_coords);
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    printf("handleRectIntersect: %f\n", time);
    // Starts the scattering site calculation
    start = std::chrono::steady_clock::now();
    findScatteringSites(int_times, int_coords);
    stop = std::chrono::steady_clock::now();
    time = std::chrono::duration<double>(stop - start).count();
    printf("findScatteringSites: %f\n", time);
    // Starts the elastic scattering calculation
    start = std::chrono::steady_clock::now();
    findScatteringVels();
    stop = std::chrono::steady_clock::now();
    time = std::chrono::duration<double>(stop - start).count();
    printf("findScatteringTimes: %f\n", time);
}
