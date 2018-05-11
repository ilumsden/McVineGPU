#include <cstdio>
#include <ctime>
#include <random>

#include <cuda_runtime.h>

#include "Intersect.cuh"

int main()
{
    std::shared_ptr<Box> b = std::make_shared<Box>(1, 1, 1);
    std::vector< std::shared_ptr<Ray> > rays;
    std::normal_distribution<double> norm(5, 2);
    std::default_random_engine re(time(NULL));
    std::uniform_real_distribution<double> vel(0, 5);
    for (int i = 0; i < (1 << 10); i++)
    {
        rays.push_back(std::make_shared<Ray>(norm(re), norm(re), norm(re), vel(re), vel(re), vel(re)));
        //printf("Ray #%i | %f %f %f %f %f %f\n", i, rays[i]->x, rays[i]->y, rays[i]->z,
        //                                        rays[i]->vx, rays[i]->vy, rays[i]->vz); 
    }
    CudaDriver cd;
    cd(b, rays);
    return 0;
}
