#include <cstdio>
#include <ctime>
#include <random>

#include <cuda_runtime.h>

#include <chrono>

#include "IsotropicScatterer.hpp"
#include "Beam.hpp"
#include "SystemVars.hpp"

typedef mcvine::gpu::composite::AbstractShape AbstractShape;
typedef mcvine::gpu::Beam Beam;
typedef mcvine::gpu::scatter::IsotropicScatterer IsotropicScatterer;
typedef mcvine::gpu::Ray Ray;

typedef mcvine::gpu::composite::Box Box;
typedef mcvine::gpu::composite::Cylinder Cylinder;
typedef mcvine::gpu::composite::Pyramid Pyramid;
typedef mcvine::gpu::composite::Sphere Sphere;

float mcvine::gpu::atten;

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
    mcvine::gpu::atten = 0.01;
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
                mcvine::gpu::atten = std::stof(flag.substr(eqind+1));
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
                fprintf(stderr, "Invalid Flag: %s\n\n", flag.c_str());
                fprintf(stderr, "./McVineGPUTest [flags]\n\nFlags:\n    -h|--help: Prints help info\n    --blockSize=#: Flag for specifying the CUDA block size. Replace # with an integer multiple of 32.\n    --atten=#: Flag for specifying the attenuation of the system. Replace # with the desired attenuation in meters.\n    --fname=_: Flag for specifying the file name for the final output file. Replace the _ with the desired file name.\n");
                return -1;
            }
        } 
        std::string sizeFlag = argv[1];
        blockSize = std::stoi(sizeFlag.substr(12));
    }
    auto start = std::chrono::steady_clock::now();
    // All calls to std::chrono are used for program timing.
    /* These lines create the AbstractShape pointer used for testing
     * each primative.
     */
    //std::shared_ptr<AbstractShape> b = std::make_shared<Box>(0.002, 0.05, 0.1);
    //std::shared_ptr<AbstractShape> b = std::make_shared<Cylinder>(0.05, 0.1);
    std::shared_ptr<AbstractShape> b = std::make_shared<Pyramid>(0.002, 0.05, 0.1);
    //std::shared_ptr<AbstractShape> b = std::make_shared<Sphere>(0.1);
    // The "rays" vector stores pointers to the rays representing neutrons.
    std::vector< std::shared_ptr<Ray> > rays;
    double x = -0.5; double y = 0; double z = 0;
    double vx = 1; double vy = 0; double vz = 0;
    if (b->type == "Pyramid")
    {
        z -= 0.05;
    }
    rays.push_back(std::make_shared<Ray>(x, y, z, vx, vy, vz));
    std::shared_ptr<Beam> beam = std::make_shared<Beam>(rays, 100000000, blockSize);
    mcvine::gpu::scatter::IsotropicScatterer iso(beam, b);
    iso.scatter();
    std::fstream fout;
    fout.open(fname, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!fout.is_open())
    {
        fprintf(stderr, "%s could not be openned.\n", fname.c_str());
        return -2;
    }
    fout << *beam;
    //beam->printAllData("test.txt");
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    printf("Total Time = %f s\n", time);
    return 0;
}
