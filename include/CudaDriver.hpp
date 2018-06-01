#ifndef _CUDA_DRIVER_H_
#define _CUDA_DRIVER_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <memory>
#include <vector>

#include "Box.hpp"
#include "Cylinder.hpp"
#include "Pyramid.hpp"
#include "Ray.hpp"

class CudaDriver
{
    public:
        CudaDriver(const std::vector< std::shared_ptr<Ray> > &rays, int bS);

        ~CudaDriver();

        void runCalculations(std::shared_ptr<AbstractShape> &b);
    private:

        void handleRectIntersect(std::shared_ptr<AbstractShape> &b, 
                                 std::vector<float> &host_time, 
                                 std::vector<float> &int_coords);

        void findScatteringSites(//std::shared_ptr<AbstractShape> &b,
                                 const std::vector<float> &int_times, 
                                 const std::vector<float> &int_coords,
                                 std::vector<float> &sites);

        float *rx, *ry, *rz, *vx, *vy, *vz;
        float *d_rx, *d_ry, *d_rz, *d_vx, *d_vy, *d_vz;
        int N;
        int blockSize, numBlocks;
};

#endif
