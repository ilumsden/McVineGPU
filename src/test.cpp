#include <cstdio>
#include <ctime>
#include <random>

#include <cuda_runtime.h>

#include <chrono>

#include "CudaDriver.hpp"
#include "SystemVars.hpp"

float atten;

int main(int argc, char **argv)
{
    /* Basic command line argument checking
     * The code supports a -h or --help flag for
     * showing how to use the executable. It also supports
     * a --blockSize= flag followed by an integer to allow the
     * user to manually specify the CUDA block size.
     */
    if (argc > 3 || (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")))
    {
        printf("./McVineGPUTest [flags]\n\nFlags:\n    -h|--help: Prints help info\n    --blockSize=#: Flag for specifying the CUDA block size. Replace # with an integer multiple of 32.\n    --atten=#: Flag for specifying the attenuation of the system. Replace # with the desired attenuation in meters.\n    --fname=_: Flag for specifying the file name for the final output file. Replace the _ with the desired file name.\n");
        return 0;
    }
    /* The default block size is 512 threads/block.
     * If the --blockSize flag is used, the value specified will
     * be used instead.
     */
    int blockSize = 512;
    atten = 0.01;
    std::string fname = "finalData.dat";
    if (argc > 1)
    {
        for (int i = 1; i < argc; i++)
        {
            std::string flag = argv[i];
            std::size_t eqind = flag.find("=");
            std::string descript = flag.substr(0, eqind);
            if (descript == "--blockSize")
            {
                blockSize = std::stoi(flag.substr(eqind+1));
            }
            else if (descript == "--atten")
            {
                atten = std::stof(flag.substr(eqind+1));
            }
            else if (descript == "--fname")
            {
                 fname = flag.substr(eqind+1);
            }
            else if (descript == "-h" || descript == "--help")
            {
                printf("./McVineGPUTest [flags]\n\nFlags:\n    -h|--help: Prints help info\n    --blockSize=#: Flag for specifying the CUDA block size. Replace # with an integer multiple of 32.\n    --atten=#: Flag for specifying the attenuation of the system. Replace # with the desired attenuation in meters.\n    --fname=_: Flag for specifying the file name for the final output file. Replace the _ with the desired file name.\n");
                return 0;
            }
            else
            {
                printf("Invalid Flag: %s\n\n", flag.c_str());
                printf("./McVineGPUTest [flags]\n\nFlags:\n    -h|--help: Prints help info\n    --blockSize=#: Flag for specifying the CUDA block size. Replace # with an integer multiple of 32.\n    --atten=#: Flag for specifying the attenuation of the system. Replace # with the desired attenuation in meters.\n    --fname=_: Flag for specifying the file name for the final output file. Replace the _ with the desired file name.\n");
                return -1;
            }
        } 
        std::string sizeFlag = argv[1];
        blockSize = std::stoi(sizeFlag.substr(12));
    }
    // All calls to std::chrono are used for program timing.
    auto start = std::chrono::steady_clock::now();
    /* These lines create the AbstractShape pointer used for testing
     * each primative.
     */
    std::shared_ptr<AbstractShape> b = std::make_shared<Box>(0.002, 0.05, 0.1);
    //std::shared_ptr<AbstractShape> b = std::make_shared<Cylinder>(0.05, 0.1);
    //std::shared_ptr<AbstractShape> b = std::make_shared<Pyramid>(0.002, 0.05, 0.1);
    //std::shared_ptr<AbstractShape> b = std::make_shared<Sphere>(0.1);
    // The "rays" vector stores pointers to the rays representing neutrons.
    std::vector< std::shared_ptr<Ray> > rays;
    double x = -0.5; double y = 0; double z = 0;
    double vx = 1; double vy = 0; double vz = 0;
    // Debugging print to stdout.
    printf("Starting data creation\n");
    auto createStart = std::chrono::steady_clock::now();
    /* This for loop randomly generates the initial ray data.
     * The interior for loop is used to ensure the neutrons are moving
     * in the general direction of the origin.
     */
    for (int i = 0; i < 100000000; i++)
    {
        printf("i = %i\n", i);
        rays.push_back(std::make_shared<Ray>(x, y, z, vx, vy, vz));
    }
    auto createStop = std::chrono::steady_clock::now();
    double createTime = std::chrono::duration<double>(createStop - createStart).count();
    printf("Data Creation: %f\n", createTime);
    // A CudaDriver object is created and used to run tests.
    auto consStart = std::chrono::steady_clock::now();
    CudaDriver cd(rays, b, blockSize);
    auto consStop = std::chrono::steady_clock::now();
    double consTime = std::chrono::duration<double>(consStop - consStart).count();
    printf("CudaDriver Constructor: %f\n", consTime);
    cd.runCalculations();
    std::fstream fout;
    fout.open(fname, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!fout.is_open())
    {
        fprintf(stderr, "%s could not be openned.\n", fname.c_str());
        return -2;
    }
    fout << cd;
    //cd.printToH5(fname);
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    printf("Total Time = %f s\n", time);
    return 0;
}
