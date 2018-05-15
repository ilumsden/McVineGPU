#include <cstdio>
#include <ctime>
#include <random>

#include <cuda_runtime.h>

#include "CudaDriver.hpp"

int main()
{
    std::shared_ptr<Box> b = std::make_shared<Box>(2, 2, 2);
    std::vector< std::shared_ptr<Ray> > rays;
    /*rays.push_back(std::make_shared<Ray>(5, 5, 5, -1.2, -1.2, -1));
    rays.push_back(std::make_shared<Ray>(3, 5, 6, -0.5, -0.6, -0.75));
    rays.push_back(std::make_shared<Ray>(-5, -5, -5, 0.6, 0.5, 0.6));*/
    std::normal_distribution<double> norm(5, 1);
    std::default_random_engine re(time(NULL));
    std::uniform_real_distribution<double> vel(0, 1);
    for (int i = 0; i < (1 << 10); i++)
    {
        std::shared_ptr<Ray> tmp = std::make_shared<Ray>(norm(re), norm(re), norm(re));
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
    CudaDriver cd;
    cd(b, rays);
    return 0;
}
