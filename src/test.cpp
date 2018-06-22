#include <cstdio>
#include <ctime>
#include <random>

#include <cuda_runtime.h>

#include <chrono>

#include "CudaDriver.hpp"

int main(int argc, char **argv)
{
    /* Basic command line argument checking
     * The code supports a -h or --help flag for
     * showing how to use the executable. It also supports
     * a --blockSize= flag followed by an integer to allow the
     * user to manually specify the CUDA block size.
     */
    if (argc > 2 || (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")))
    {
        printf("./McVineGPUTest [-h|--help] [--blockSize=]\n");
        return 0;
    }
    /* The default block size is 512 threads/block.
     * If the --blockSize flag is used, the value specified will
     * be used instead.
     */
    int blockSize = 512;
    if (argc == 2)
    {
        std::string sizeFlag = argv[1];
        blockSize = std::stoi(sizeFlag.substr(12));
    }
    // All calls to std::chrono are used for program timing.
    auto start = std::chrono::steady_clock::now();
    /* These lines create the AbstractShape pointer used for testing
     * each primative.
     */
    std::shared_ptr<AbstractShape> b = std::make_shared<Box>(2, 2, 2);
    //std::shared_ptr<AbstractShape> b = std::make_shared<Cylinder>(2, 5);
    //std::shared_ptr<AbstractShape> b = std::make_shared<Pyramid>(4, 4, 4);
    //std::shared_ptr<AbstractShape> b = std::make_shared<Sphere>(3);
    // The "rays" vector stores pointers to the rays representing neutrons.
    std::vector< std::shared_ptr<Ray> > rays;
    /* These distributions are used for setting the ray positions for
     * different tests. Testing for Pyramid solids uses the x, y, and z
     * distributions. All others use norm.
     */
    std::normal_distribution<double> norm(5, 1);
    //std::normal_distribution<double> x(2, 1);
    //std::normal_distribution<double> y(-2, 1);
    //std::normal_distribution<double> z(-5, 1);
    std::default_random_engine re(time(NULL));
    // "vel" is the distribution used for setting ray velocities.
    std::uniform_real_distribution<double> vel(0, 1);
    // Debugging print to stdout.
    printf("Starting data creation\n");
    auto createStart = std::chrono::steady_clock::now();
    /* This for loop randomly generates the initial ray data.
     * The interior for loop is used to ensure the neutrons are moving
     * in the general direction of the origin.
     */
    for (int i = 0; i < 1000000; i++)
    {
        printf("i = %i\n", i);
        //std::shared_ptr<Ray> tmp = std::make_shared<Ray>(x(re), y(re), z(re));
        std::shared_ptr<Ray> tmp = std::make_shared<Ray>(norm(re), norm(re), norm(re));
        double veltmp[3];
        for (int i = 0; i < 3; i++)
        {
            double velo = vel(re);
            double pos = (i == 0) ? tmp->origin[0] : ((i == 1) ? tmp->origin[1] : tmp->origin[2]);
            if ((pos > 0 && velo > 0) || (pos < 0 && velo < 0))
            {
                velo *= -1;
            }
            veltmp[i] = velo;
        }
        tmp->setVelocities(veltmp[0], veltmp[1], veltmp[2]);
        rays.push_back(tmp);
    }
    auto createStop = std::chrono::steady_clock::now();
    double createTime = std::chrono::duration<double>(createStop - createStart).count();
    printf("Data Creation: %f\n", createTime);
    // A CudaDriver object is created and used to run tests.
    auto consStart = std::chrono::steady_clock::now();
    CudaDriver cd(rays, blockSize);
    auto consStop = std::chrono::steady_clock::now();
    double consTime = std::chrono::duration<double>(consStop - consStart).count();
    printf("CudaDriver Constructor: %f\n", consTime);
    cd.runCalculations(b);
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    printf("Total Time = %f s\n", time);
    return 0;
}
