#include <cstdio>
#include <ctime>
#include <random>

#include <cuda_runtime.h>

#include <chrono>

#include "CudaDriver.hpp"

int main(int argc, char **argv)
{
    if (argc > 2 || (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")))
    {
        printf("./McVineGPUTest [-h|--help] [--blockSize=]\n");
        return 0;
    }
    int blockSize = 512;
    if (argc == 2)
    {
        std::string sizeFlag = argv[1];
        blockSize = std::stoi(sizeFlag.substr(12));
    }
    auto start = std::chrono::steady_clock::now();
    //std::shared_ptr<AbstractShape> b = std::make_shared<Box>(2, 2, 2);
    //std::shared_ptr<AbstractShape> b = std::make_shared<Cylinder>(2, 5);
    std::shared_ptr<AbstractShape> b = std::make_shared<Pyramid>(4, 4, 4);
    std::vector< std::shared_ptr<Ray> > rays;
    /*rays.push_back(std::make_shared<Ray>(5, 5, 5, -1.2, -1.2, -1));
    rays.push_back(std::make_shared<Ray>(3, 5, 6, -0.5, -0.6, -0.75));
    rays.push_back(std::make_shared<Ray>(-5, -5, -5, 0.6, 0.5, 0.6));*/
    //std::normal_distribution<double> norm(5, 1);
    std::normal_distribution<double> x(2, 1);
    std::normal_distribution<double> y(-2, 1);
    std::normal_distribution<double> z(-5, 1);
    std::default_random_engine re(time(NULL));
    std::uniform_real_distribution<double> vel(0, 1);
    printf("Starting data creation\n");
    auto createStart = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000000; i++)
    {
        printf("i = %i\n", i);
        std::shared_ptr<Ray> tmp = std::make_shared<Ray>(x(re), y(re), z(re));
        double veltmp[3];
        for (int i = 0; i < 3; i++)
        {
            double velo = vel(re);
            double pos = (i == 0) ? tmp->x : ((i == 1) ? tmp->y : tmp->z);
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
