#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <limits>
#include <fstream>
//#include "H5Cpp.h"

#if defined(RANDTEST)
#include <cmath>
#endif

#include <chrono>

#include "CudaDriver.hpp"

CudaDriver::CudaDriver(std::vector< std::shared_ptr<Ray> > &rays, 
                       std::shared_ptr<AbstractShape> &shape, int bS)
{ 
    N = (int)(rays.size());
    b = shape;
    CudaErrchk( cudaGetDeviceCount(&nGpu) );
    int jump = N / nGpu;
    int rem = N % nGpu;
    for (int i = 0; i <= nGpu; i++)
    {
        steps.push_back(i * jump);
    }
    for (int i = 1; i <= rem; i++)
    {
        steps[i]++;
        for (int j = i+1; j < (int)(steps.size()); j++)
        {
            steps[j]++;
        }
    }
    blockNums.resize(nGpu);
    for (int i = 0; i < (int)(steps.size())-1; i++)
    {
        blockNums[i] = ((steps[i+1] - steps[i]) + blockSize - 1) / blockSize;
    }
    // Calculates the CUDA launch parameters using bS
    /*blockSize = bS;
    numBlocks = (N + blockSize - 1) / blockSize;
    printf("CUDA Launch Params: %i, %i\n", numBlocks, blockSize);*/
    /* Allocates both host and device memory for the float arrays that
     * will be used to store the data passed to the CUDA functions.
     */
    rayptr = &rays;
    origins = (Vec3<float>*)malloc(N*sizeof(Vec3<float>));
    //CudaErrchk( cudaMalloc(&d_origins, N*sizeof(Vec3<float>)) );
    vel = (Vec3<float>*)malloc(N*sizeof(Vec3<float>));
    //CudaErrchk( cudaMalloc(&d_vel, N*sizeof(Vec3<float>)) );
    times = (float*)malloc(N*sizeof(float));
    //CudaErrchk( cudaMalloc(&d_times, N*sizeof(float)) );
    probs = (float*)malloc(N*sizeof(float));
    //CudaErrchk( cudaMalloc(&d_probs, N*sizeof(float)) );
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
    for (int i = 0; i < nGpu; i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(d_origins[i]), (steps[i+1]-steps[i])*sizeof(Vec3<float>)) );
        CudaErrchk( cudaMemcpy(d_origins[i], &(origins[steps[i]]), (steps[i+1]-steps[i])*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
        CudaErrchk( cudaMalloc(&(d_vel[i]), (steps[i+1]-steps[i])*sizeof(Vec3<float>)) );
        CudaErrchk( cudaMemcpy(d_vel[i], &(vel[steps[i]]), (steps[i+1]-steps[i])*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
        CudaErrchk( cudaMalloc(&(d_times[i]), (steps[i+1]-steps[i])*sizeof(float)) );
        CudaErrchk( cudaMemcpy(d_times[i], &(times[steps[i]]), (steps[i+1]-steps[i])*sizeof(float), cudaMemcpyHostToDevice) );
        CudaErrchk( cudaMalloc(&(d_probs[i]), (steps[i+1]-steps[i])*sizeof(float)) );
        CudaErrchk( cudaMemcpy(d_probs[i], &(probs[steps[i]]), (steps[i+1]-steps[i])*sizeof(float), cudaMemcpyHostToDevice) );
    }
    // Copies the data from the host arrays to the device arrays.
    //CudaErrchk( cudaMemcpy(d_origins, origins, N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
    //CudaErrchk( cudaMemcpy(d_vel, vel, N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
    //CudaErrchk( cudaMemcpy(d_times, times, N*sizeof(float), cudaMemcpyHostToDevice) );
    //CudaErrchk( cudaMemcpy(d_probs, probs, N*sizeof(float), cudaMemcpyHostToDevice) );
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

void CudaDriver::printFullData(const std::string &fname)
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

void CudaDriver::updateRays()
{
    for (int i = 0; i < N; i++)
    {
        (*rayptr)[i]->update(origins[i], vel[i], times[i], probs[i]);
    }
}

/*void CudaDriver::printToH5(const std::string &fname, const bool endianness)
{
    using namespace H5;
    if (fname.substr(fname.rfind(".")) != ".h5")
    {
        fprintf(stderr, "%s is not a supported HDF5 file name.\n", fname.c_str());
        return;
    }
    float data[N][4];
    for (int i = 0; i < N; i++)
    {
        data[i][0] = vel[i][0];
        data[i][1] = vel[i][1];
        data[i][2] = vel[i][2];
        data[i][3] = probs[i];
    }
    try
    {
        PredType datatype = PredType::IEEE_F32LE;
        if (endianness)
        {
            datatype = PredType::IEEE_F32BE;
        }
        Exception::dontPrint();
        H5File file(fname, H5F_ACC_TRUNC);
        hsize_t dims[2] = {N, 4};
        DataSpace dspace(2, dims);
        DataSet dset = file.createDataSet("VecProbData", datatype, dspace); 
        dset.write(data, datatype);
    }
    catch(FileIException error)
    {
        error.printErrorStack();
        //H5Eprint(stderr);
        exit(-3);
    }
    catch(DataSetIException error)
    {
        error.printErrorStack();
        //H5Eprint(stderr);
        exit(-3);
    }
    catch(DataSpaceIException error)
    {
        error.printErrorStack();
        //H5Eprint(stderr);
        exit(-3);
    }
    return;
}*/

void CudaDriver::handleExteriorIntersect(std::vector<float> &host_time,
                                         std::vector< Vec3<float> > &int_coords)
{
    /* Calls the shape's intersect function.
     * Inheritance is used to choose the correct algorithm for intersection.
     */
    b->exteriorIntersect(d_origins, d_vel, N, blockSize, numBlocks, host_time, int_coords);
    // Opens a file stream and prints the relevant data to time.txt
    // NOTE: this is for debugging purposes only. This will be removed later.
#if defined(DEBUG) || defined(PRINT1)
    std::fstream fout;
    fout.open("time.txt", std::ios::out);
    if (!fout.is_open())
    {
        std::cerr << "time.txt could not be opened.\n";
        exit(-1);
    }
    for (int i = 0; i < (int)(int_coords.size()); i++)
    {
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
    fout.close();
#endif
    return;
}

void CudaDriver::findScatteringSites(const std::vector<float> &int_times, 
                                     const std::vector< Vec3<float> > &int_coords)
{
#if defined(DEBUG) || defined(PRINT2)
    std::vector< Vec3<float> > tmp;
    tmp.resize(N);
    Vec3<float> *ta = tmp.data();
    memcpy(ta, origins, N*sizeof(Vec3<float>));
#endif
    // Stores the size of the `int_times` for later
    int tsize = (int)(int_times.size());
    /* Allocates memory for a device-side array that stores the
     * data passed in from `int_times`.
     */
    /*float *ts;
    CudaErrchk( cudaMalloc(&ts, 2*N*sizeof(float)) );
    CudaErrchk( cudaMemcpy(ts, int_times.data(), 2*N*sizeof(float), cudaMemcpyHostToDevice) );*/
    std::vector<float*> ts;
    ts.resize(nGpu);
    for (int i = 0; i < nGpu; i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(ts[i]), 2*(steps[i+1]-steps[i])*sizeof(float)) );
        CudaErrchk( cudaMemcpy(ts[i], &(int_times.data()[steps[i]]), 2*(steps[i+1]-steps[i])*sizeof(float), cudaMemcpyHostToDevice) );
    }
    /* `pos` is a device-side array that stores the coordinates of the
     * scattering sites for the neutrons.
     * The default value of its data is FLT_MAX.
     */
    /*Vec3<float> *pos;
    CudaErrchk( cudaMalloc(&pos, N*sizeof(Vec3<float>)) );
    initArray< Vec3<float> ><<<numBlocks, blockSize>>>(pos, N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
    CudaErrchkNoCode();*/
    std::vector<Vec3<float>*> pos;
    pos.resize(nGpu);
    for (int i = 0; i < nGpu; i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(pos[i]), (steps[i+1]-steps[i])*sizeof(Vec3<float>)) );
        initArray< Vec3<float> ><<<blockNums[i], blockSize>>>(pos[i], (steps[i+1]-steps[i]), Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX));
        CudaErrchkNoCode();
    }
    /* `scat_times` is a device_side array that stores the times at
     * which the neutrons reach their scattering sites.
     * The default value of its data is -5.
     */
    /*float *scat_times;
    CudaErrchk( cudaMalloc(&scat_times, N*sizeof(float)) );
    initArray<float><<<numBlocks, blockSize>>>(scat_times, N, -5);*/
    std::vector<Vec3<float>*> scat_times;
    pos.resize(nGpu);
    for (int i = 0; i < nGpu; i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(scat_times[i]), (steps[i+1]-steps[i])*sizeof(float)) );
        initArray< Vec3<float> ><<<blockNums[i], blockSize>>>(scat_times[i], (steps[i+1]-steps[i]), -5);
        CudaErrchkNoCode();
    }
    /*curandGenerator_t gen;
    float *d_randnums;
    CudaErrchk( cudaMalloc(&d_randnums, N*sizeof(float)) );
    CuRandErrchk( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    CuRandErrchk( curandSetPseudoRandomGeneratorSeed(gen, time(NULL)) );
    CuRandErrchk( curandGenerateUniform(gen, d_randnums, N) );*/
    std::vector<float*> d_randnums;
    d_randnums.resize(nGpu);
    for (int i = 0; i < nGpu; i++)
    {
        CudaErrchk( cudaSetDevice(i) );
        CudaErrchk( cudaMalloc(&(d_randnums[i]), (steps[i+1]-steps[i])*sizeof(float)) );
        curandGenerator_t gen;
        CuRandErrchk( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
        CuRandErrchk( curandSetPseudoRandomGeneratorSeed(gen, time(NULL)) );
        CuRandErrchk( curandGenerateUniform(gen, d_randnums[i], (steps[i+1]-steps[i])) );
        CuRandErrchk( curandDestroyGenerator(gen) );
    }
    // Calls the kernel for determining the scattering sites for the neutrons
    //calcScatteringSites<<<numBlocks, blockSize>>>(ts, d_origins, d_vel, pos, scat_times, d_randnums, N);
    /* Propagates the neutrons to their scattering sites.
     * In other words, the scattering coordinates and times are copied
     * into the device arrays that store the neutrons' origins and times
     * (d_origins and d_times respectively).
     */
    //propagate<<<numBlocks, blockSize>>>(d_origins, d_times, pos, scat_times, N);
    //CudaErrchkNoCode();
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
    updateProbability<<<numBlocks, blockSize>>>(d_probs, d_origins, ic, 1, 2, atten, N);
    CudaErrchkNoCode();
    // Updates the host-side arrays for the edited neutron data.
    CudaErrchk( cudaMemcpy(origins, d_origins, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(times, d_times, N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(probs, d_probs, N*sizeof(float), cudaMemcpyDeviceToHost) );
#if defined(DEBUG) || defined(PRINT2)
    std::fstream fout;
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
    fout.close();
#endif
    // Frees the device memory allocated above.
    CudaErrchk( cudaFree(ts) );
    CudaErrchk( cudaFree(ic) );
    CudaErrchk( cudaFree(pos) );
    CudaErrchk( cudaFree(d_randnums) );
    return;
}

void CudaDriver::findScatteringVels()
{
#if defined(DEBUG) || defined(PRINT3)
    std::vector< Vec3<float> > tmp;
    tmp.resize(N);
    Vec3<float> *ta = tmp.data();
    memcpy(ta, vel, N*sizeof(Vec3<float>));
#endif
#if defined(DEBUG) || defined(RANDTEST)
    std::vector<float> thetas, phis;
#endif
    curandGenerator_t gen;
    float *d_randnums;
    CudaErrchk( cudaMalloc(&d_randnums, 2*N*sizeof(float)) );
    CuRandErrchk( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    CuRandErrchk( curandSetPseudoRandomGeneratorSeed(gen, time(NULL)) );
    CuRandErrchk( curandGenerateUniform(gen, d_randnums, 2*N) );
    /* Calls the elasticScatteringKernel function to update the neutron
     * velocities post-elastic scattering.
     */
    /*isotropicScatteringKernel<<<numBlocks, blockSize>>>(d_times,
                                                        d_vel,
                                                        d_randnums, N);*/
    scatter<<<numBlocks, blockSize>>>(0, d_times, d_vel, d_randnums, N);
    CudaErrchkNoCode();
    /* Copies the new neutron velocities into the host-side neutron
     * velocity array.
     */
    CudaErrchk( cudaMemcpy(vel, d_vel, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    // Opens a file stream and prints the 
    // relevant data to scatteringVels.txt
#if defined(DEBUG) || defined(PRINT3)
    std::fstream fout;
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
    fout.close();
#endif
#if defined(DEBUG) || defined(RANDTEST)
    for (int i = 0; i < N; i++)
    {
        thetas.push_back(acos(vel[i][2] / vel[i].length()));
        phis.push_back(atan2(vel[i][1], vel[i][0]));
    }
    std::sort(thetas.begin(), thetas.end());
    std::sort(phis.begin(), phis.end());
    std::fstream f1, f2;
    f1.open("thetas.txt", std::ios::out);
    if (!f1.is_open())
    {
        std::cerr << "thetas.txt could not be openned.\n";
        exit(-2);
    }
    f2.open("phis.txt", std::ios::out);
    if (!f2.is_open())
    {
        std::cerr << "phis.txt could not be openned.\n";
        exit(-2);
    }
    f1 << "Theta Values (Radians): Should range from 0 to Pi\n";
    f2 << "Phi Values (Radians): Should range from 0 to 2*Pi\n";
    for (int i = 0; i < (int)(thetas.size()); i++)
    {
        f1 << thetas[i] << "\n";
        f2 << phis[i] << "\n";
    }
    f1.close();
    f2.close();
#endif
    CuRandErrchk( curandDestroyGenerator(gen) );
    CudaErrchk( cudaFree(d_randnums) );
}

void CudaDriver::handleInteriorIntersect()
{
#if defined(DEBUG) || defined(PRINT4)
    std::vector< Vec3<float> > tmp;
    tmp.resize(N);
    Vec3<float> *ta = tmp.data();
    memcpy(ta, origins, N*sizeof(Vec3<float>));
#endif
    std::vector<float> int_times;
    std::vector< Vec3<float> > int_coords;
    b->interiorIntersect(d_origins, d_vel, N, blockSize, numBlocks, int_times, int_coords); 
    float *exit_times;
    CudaErrchk( cudaMalloc(&exit_times, N*sizeof(float)) );
    CudaErrchk( cudaMemcpy(exit_times, int_times.data(), N*sizeof(float), cudaMemcpyHostToDevice) );
    Vec3<float> *exit_coords;
    CudaErrchk( cudaMalloc(&exit_coords, N*sizeof(Vec3<float>)) );
    CudaErrchk( cudaMemcpy(exit_coords, int_coords.data(), N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
    updateProbability<<<numBlocks, blockSize>>>(d_probs, exit_coords, d_origins, 1, 1, atten, N);
    propagate<<<numBlocks, blockSize>>>(d_origins, d_times, exit_coords, exit_times, N);
    CudaErrchk( cudaMemcpy(origins, d_origins, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(times, d_times, N*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(probs, d_probs, N*sizeof(float), cudaMemcpyDeviceToHost) );
#if defined(DEBUG) || defined(PRINT4)
    std::fstream fout;
    fout.open("exit.txt", std::ios::out);
    if (!fout.is_open())
    {
        std::cerr << "exit.txt could not be openned.\n";
        exit(-2);
    }
    for (int i = 0; i < N; i++)
    {
        std::string buf = "        ";
        fout << "\n";
        fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
             << tmp[i][0] << " " << tmp[i][1] << " " << tmp[i][2] << " || "
             << vel[i][0] << " " << vel[i][1] << " " << vel[i][2] << " | "
             << origins[i][0] << "\n";
        fout << buf << " " << buf << " " << buf << "    "
             << buf << " " << buf << " " << buf << "   "
             << std::fixed << std::setprecision(5) << std::setw(8) << std::right
             << origins[i][1] << "\n";
        fout << buf << " " << buf << " " << buf << "    "
             << buf << " " << buf << " " << buf << "   "
             << std::fixed << std::setprecision(5) << std::setw(8) << std::right
             << origins[i][2] << "\n";
        fout << buf << " " << buf << " " << buf << "    "
             << buf << " " << buf << " " << buf << "   "
             << std::fixed << std::setprecision(5) << std::setw(8) << std::right
             << times[i] << "\n";
        fout << buf << " " << buf << " " << buf << "    "
             << buf << " " << buf << " " << buf << "   "
             << std::fixed << std::setprecision(5) << std::setw(8) << std::right
             << probs[i] << "\n";
    }
    fout.close();
#endif
    CudaErrchk( cudaFree(exit_times) );
    CudaErrchk( cudaFree(exit_coords) );
}

void CudaDriver::runCalculations()
{
    /* Creates the vectors that will store the intersection
     * times and coordinates.
     */
    std::vector<float> int_times;
    std::vector< Vec3<float> > int_coords;
    // Starts the intersection calculation
    handleExteriorIntersect(int_times, int_coords);
    // Starts the scattering site calculation
    findScatteringSites(int_times, int_coords);
    // Starts the elastic scattering calculation
    findScatteringVels();
    handleInteriorIntersect();
}

std::ostream& operator<<(std::ostream &fout, const CudaDriver &cd)
{
    std::vector<float> data;
    for (int i = 0; i < cd.N; i++)
    {
        data.push_back(cd.vel[i][0]);
        data.push_back(cd.vel[i][1]);
        data.push_back(cd.vel[i][2]);
        data.push_back(cd.probs[i]);
    }
    char *bytes = (char*)(data.data());
    fout.write(bytes, ((int)(data.size()))*sizeof(float));
    return fout;
}
