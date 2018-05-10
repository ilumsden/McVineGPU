#include <thrust/host_vector.h>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <random>
#include <vector>

#include "Box.hpp"
#include "Intersect.cu"
#include "Ray.hpp"

int main()
{
    Box *b = new Box(1, 1, 1);
    std::vector<Ray*> rays;
    std::normal_distribution<double> norm(5, 2);
    std::default_random_engine re(time(NULL));
    std::uniform_real_distribution<double> vel(0, 5);
    for (int i = 0; i < (1 << 10); i++)
    {
        rays.push_back(new Ray(norm(re), norm(re), norm(re), vel(re), vel(re), vel(re)));
        //printf("Ray #%i | %f %f %f %f %f %f\n", i, rays[i]->x, rays[i]->y, rays[i]->z,
        //                                        rays[i]->vx, rays[i]->vy, rays[i]->vz); 
    }
    cudaHandler(b, rays);
    return 0;
}
